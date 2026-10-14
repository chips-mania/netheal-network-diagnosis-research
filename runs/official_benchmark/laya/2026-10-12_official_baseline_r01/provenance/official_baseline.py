"""Base LAYA in the frozen Jev v4 episode loop; no API clients or training."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import asyncio
import collections
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / 'netheal-topology-generalization/scripts'
sys.path.insert(0, str(LEGACY))
import jev_official_eval_v4 as env
from jev_context_codec_v4 import encode_checked, dumps
from jev_tree_partition_v4 import restore

BASE = ROOT / '.runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148'
OUT = ROOT / 'runs/official_benchmark/laya/2026-10-12_official_baseline_r01'
SPEC = ROOT / 'docs/specs/laya_official_baseline_execution_20261012.md'
MAX_LEN = 8192
HEAD_LEN = 8192


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(x):
    if isinstance(x, dict):
        assert not set(x) & {'ground_truth_fault', 'ground_truth', 'injected_fault', 'fault_info', 'correct_diagnosis', 'expected_solution'}
        for v in x.values():
            audit(v)
    elif isinstance(x, list):
        for v in x:
            audit(v)


class Backend:
    def __init__(self):
        import torch
        import laya.train as train
        import laya.common as common
        self.torch, self.train, self.common = torch, train, common
        torch.set_num_threads(4)
        torch.manual_seed(1012)
        # Native TransformerEncoder fastpath materializes quadratic attention weights.
        # The SDPA path avoids that allocation without changing weights or input.
        torch.backends.mha.set_fastpath_enabled(False)
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(.88)
        self.model, self.tok, self.cfg = train.load_checkpoint(str(BASE))
        self.model.to('cuda').eval()
        torch.cuda.reset_peak_memory_stats()

    def encode(self, text, q):
        internal = self.train.to_internal('decision', q['decision'])
        # Reject even distinguishable truncated candidates/instructions, not just dropped markers.
        head, markers, stats = self.common.build_head(self.tok, internal, HEAD_LEN)
        full_head, full_markers, _ = self.common.build_head(self.tok, internal, 100000)
        assert head == full_head and markers == full_markers, 'Head truncation'
        for option in self.common.render_options(internal):
            assert len(self.common._encode_question_text(self.tok, ' ' + option.replace(self.tok.mask_token, ' '), add_special_tokens=False)) <= 48, 'Option exceeds installed rendering limit'
        ids, markers, stats, cut = self.common.build_sequence(self.tok, text, internal, MAX_LEN, HEAD_LEN,
                  return_stats=True, return_truncation_stats=True)
        ok = not cut['truncated'] and len(ids) <= MAX_LEN and len(markers) == stats['options_distinct'] == len(q['decision']['criteria'])
        return ok, {'ids': ids, 'markers': markers, 'qtype': 0}

    def partitions(self, state, purpose, options, deadline):
        audit(state)
        q = env.question(purpose, options)
        assert 1 <= len(options) <= 255
        overhead = len(json.dumps(q, ensure_ascii=False).encode()) + 512
        cache = {}

        def inspect(s):
            if time.monotonic() >= deadline:
                raise TimeoutError('episode deadline during partitioning')
            key = dumps(s)
            if key not in cache:
                text = encode_checked(s)
                if len(text.encode()) + overhead > 24000:
                    cache[key] = (False, text, None)
                else:
                    ok, encoded = self.encode(text, q)
                    cache[key] = (ok, text, encoded)
            return cache[key]

        # Preserve Jev's exact envelopes, boundaries and weights whenever they fit LAYA.
        original = env.partition_state(state, purpose, options)
        if all(inspect(c)[0] for c in original):
            chunks = original
            mode = 'jev_v4_exact'
        else:
            identity = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
            common = {k: state[k] for k in ('task_description', 'objective', 'hint', 'network_size', 'max_steps', 'solver_turn') if k in state}

            def wrap(items):
                return dict(common, partition_notice='Lossless evidence partition: path is the location in the original JSON state; value is the exact subtree. All partitions together preserve all evidence. Evaluate the same options from this evidence.',
                    original_state_sha256=identity, evidence_fragments=items,
                    partition_weight=sum(len(dumps(x['value']).encode()) for x in items))
            assert inspect(wrap([]))[0], 'Fixed question exceeds context budget'
            fragments = []

            def split(value, path):
                item = {'path': path, 'value': value}
                if inspect(wrap([item]))[0]:
                    fragments.append(item)
                elif isinstance(value, dict) and value:
                    for k, v in value.items():
                        split(v, path + [k])
                elif isinstance(value, list) and value:
                    for i, v in enumerate(value):
                        split(v, path + [i])
                else:
                    raise ValueError('Indivisible scalar exceeds context')
            split(state, [])
            assert dumps(restore(fragments)) == dumps(state)
            chunks, current = [], []
            for f in fragments:
                if current and not inspect(wrap(current + [f]))[0]:
                    chunks.append(wrap(current))
                    current = []
                current.append(f)
            if current:
                chunks.append(wrap(current))
            mode = 'laya_token_tree'
        weights = [1] if len(chunks) == 1 else [c.get('partition_weight', len(c.get('history', []))) for c in chunks]
        assert all(w > 0 for w in weights)
        encoded = [inspect(c) for c in chunks]
        assert all(ok for ok, _, _ in encoded)
        return q, chunks, weights, encoded, mode

    def forward(self, encoded, count):
        torch = self.torch
        batch = self.common.collate_items([[encoded]], self.tok.pad_token_id)
        temperature = self.cfg.get('temperature_by_options', {}).get(self.common.temp_bucket(0, count), self.cfg['temperature'][0])
        temperature = self.common.clamp_temperature(temperature)
        with torch.inference_mode():
            logits = self.train._forward(self.model, batch, torch.device('cuda'), False, False)[0, :count]
            p = torch.softmax(logits / temperature, dim=-1).cpu().tolist()
        assert len(p) == count and all(0 <= v <= 1 for v in p) and abs(sum(p) - 1) < 1e-5
        return p, temperature


class Laya:
    backend = None

    def __init__(self, unused_key, folder, deadline):
        self.folder, self.deadline, self.calls, self.questions = folder, deadline, 0, 0
        self.client = self  # Frozen episode's finally awaits client.aclose(). No network client exists.

    async def aclose(self):
        pass

    async def choose(self, state, purpose, options):
        self.questions += 1
        start = time.monotonic()
        q, chunks, weights, inputs, mode = self.backend.partitions(state, purpose, options, self.deadline)
        probabilities = []
        for chunk, weight, (_, text, encoded) in zip(chunks, weights, inputs):
            if time.monotonic() >= self.deadline:
                raise TimeoutError('episode deadline before forward')
            self.calls += 1
            record = {'question': self.questions, 'call': self.calls, 'request': {'state': text, 'questions': q},
                      'mode': mode, 'weight': weight, 'tokens': len(encoded['ids']),
                      'raw_state_sha256': hashlib.sha256(json.dumps(chunk, sort_keys=True).encode()).hexdigest()}
            env.append(self.folder / 'request_started.jsonl', {'call': self.calls, 'question': self.questions, 'unix': time.time()})
            p, temp = self.backend.forward(encoded, len(options))
            probabilities.append(p)
            record['response'] = {'answers': {'decision': {'choice': f'c{max(range(len(p)), key=p.__getitem__):03d}',
                                     'probabilities': {f'c{i:03d}': v for i, v in enumerate(p)}}}}
            record['temperature'] = temp
            env.append(self.folder / 'requests.jsonl', record)
            if time.monotonic() >= self.deadline:
                raise TimeoutError('episode deadline after forward')
        scores = [sum(w * p[i] for w, p in zip(weights, probabilities)) / sum(weights) for i in range(len(options))]
        index = max(range(len(options)), key=scores.__getitem__)
        env.append(self.folder / 'decisions.jsonl', {'question': self.questions, 'purpose': purpose, 'mode': mode,
              'chunks': len(chunks), 'weights': weights, 'probabilities': scores, 'selected': options[index],
              'seconds': time.monotonic() - start, 'solver_turn': state['solver_turn']})
        print(json.dumps({'event': 'decision', 'turn': state['solver_turn'], 'question': self.questions,
              'purpose': purpose, 'chunks': len(chunks), 'seconds': round(time.monotonic() - start, 2)}), flush=True)
        return options[index]


def source_hashes():
    import laya.train, laya.common, laya.agent
    paths = [Path(__file__), SPEC, LEGACY / 'jev_official_eval_v4.py', LEGACY / 'jev_context_codec_v4.py',
             LEGACY / 'jev_tree_partition_v4.py', Path(laya.train.__file__), Path(laya.common.__file__), Path(laya.agent.__file__)]
    paths += [p for p in BASE.rglob('*') if p.is_file()]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def report(snapshots):
    rows = [read(OUT / 'episodes' / s['snapshot_id'] / 'result.json') for s in snapshots if (OUT / 'episodes' / s['snapshot_id'] / 'result.json').exists()]
    evaluator = env.CompetitionEvaluator()
    for r in rows:
        evaluator.add_episode_metrics(env.EpisodeMetrics(**r['metrics']))
    official = evaluator.compute_summary()
    env.save(OUT / 'summary.json', {'official_summary': official, 'episodes': rows})
    env.save(OUT / 'manifest.json', {'status': 'completed' if len(rows) == 100 else 'running', 'completed': len(rows), 'total': 100, 'updates': 0, 'api_calls': 0})
    return rows, official


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'evaluate', 'report'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ss = env.snapshots()
    if args.mode == 'report':
        report(ss)
        return
    if args.mode == 'evaluate':
        frozen = read(OUT / 'frozen_protocol.json')
        assert frozen['sources'] == source_hashes()
        assert frozen['snapshots'] == hashlib.sha256(dumps(ss).encode()).hexdigest()
        assert read(OUT / 'preflight.json')['status'] == 'passed'
    backend = Backend()
    Laya.backend = backend
    # Reuse the entire frozen loop (including candidate order, helpers, budget and metrics).
    # Only its model-client constructor changes, in this process; source on disk is untouched.
    env.Jev = Laya
    if args.mode == 'preflight':
        ref = read(env.UP / 'manikyabard-20260201-100940.json')['results'][0]
        reference = {r['snapshot_id']: r for a in ref['artifacts'] if a['label'] == 'aaa_metrics' for r in a['data']['results']}
        assert len(ss) == len(reference) == 100
        for s in ss:
            e, w, m = env.runtime(s)
            assert w._trace.scenario_fingerprint == reference[s['snapshot_id']]['scenario_fingerprint']
            assert e.max_episode_steps == 100
            e.close()
        dev = read(env.OUT / 'development_snapshot.json')
        assert dev['snapshot_id'] not in reference
        attempt_id = 0
        while (OUT / f'smoke_{attempt_id:02d}').exists():
            attempt_id += 1
        status = await env.episode(dev, None, OUT / f'smoke_{attempt_id:02d}', max_turns=3)
        assert not status.startswith('error'), status
        # Long input from development data, only public pre-diagnosis history.
        raw = ROOT / 'netheal-topology-generalization/results/laya_official_20261012_v1'
        cases = [read(p) for p in (raw / 'cases').glob('9402*.json')]
        c = max(cases, key=lambda c: len(dumps(c['trace'])))
        history = c['trace'][:next(i for i, x in enumerate(c['trace']) if x['tool'] == 'submit_diagnosis')]
        state = dict(task_description=env.TASK, objective=env.OBJECTIVE, hint=c['hint'],
              network_size=history[0]['result']['info']['network_size'], max_steps=100,
              tool_schemas=read(env.OUT / 'tool_schemas.json'), history=history, solver_turn=len(history))
        folder = OUT / f'long_smoke_{attempt_id:02d}'
        folder.mkdir(exist_ok=False)
        client = Laya(None, folder, time.monotonic() + 1200)
        await client.choose(state, 'Select the next tool.', env.TOOLS)
        # Verify maximum public parameter head on a separate development catalog.
        params = [a['parameters'] for a in c['catalog'] if env.action_tool(a) == 'ping']
        if len(params) > 255:
            params = [p for p in params if p['source'] == params[0]['source']]
        await client.choose(state, 'Select arguments for ping.', params)
        env.save(OUT / 'preflight.json', {'status': 'passed', 'fingerprints': 100, 'smoke_status': status,
                 'long_case_seed': c['seed'], 'long_smoke_calls': client.calls,
                 'gpu_peak_allocated': backend.torch.cuda.max_memory_allocated(),
                 'gpu_peak_reserved': backend.torch.cuda.max_memory_reserved()})
        sources = source_hashes()
        provenance = OUT / 'provenance'
        provenance.mkdir(exist_ok=True)
        for name in sources:
            p = ROOT / name
            if p.suffix in ('.py', '.md', '.json') and 'tokenizer' not in p.parts:
                shutil.copyfile(p, provenance / p.name)
        env.save(OUT / 'frozen_protocol.json', {'sources': sources, 'snapshots': hashlib.sha256(dumps(ss).encode()).hexdigest(),
                 'max_len': MAX_LEN, 'head_max_len': HEAD_LEN, 'seed': 1012, 'dtype': 'float32',
                 'temperature': 'checkpoint built-in clamped option-count buckets', 'concurrency': 1,
                 'retry': 'no automatic retries; interrupted attempts preserved and restarted', 'updates': 0})
        print('PREFLIGHT PASSED', flush=True)
        return
    for s in ss:
        parent = OUT / 'episodes' / s['snapshot_id']
        if (parent / 'result.json').exists():
            continue
        parent.mkdir(parents=True, exist_ok=True)
        i = 0
        while (parent / f'attempt_{i:02d}').exists():
            old = parent / f'attempt_{i:02d}'
            if (old / 'result.json').exists():
                shutil.copyfile(old / 'result.json', parent / 'result.json')
                break
            env.save(old / 'interruption.json', {'status': 'interrupted_preserved', 'resumed_unix': time.time()})
            i += 1
        else:
            attempt = parent / f'attempt_{i:02d}'
            await env.episode(s, None, attempt)
            shutil.copyfile(attempt / 'result.json', parent / 'result.json')
        rows, _ = report(ss)
        print('PROGRESS', len(rows), '/100', flush=True)
    env.save(OUT / 'resources.json', {'gpu_peak_allocated': backend.torch.cuda.max_memory_allocated(),
              'gpu_peak_reserved': backend.torch.cuda.max_memory_reserved(), 'gpu': backend.torch.cuda.get_device_name(),
              'torch': backend.torch.__version__, 'python': sys.version})
    print('EVALUATION COMPLETE', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
