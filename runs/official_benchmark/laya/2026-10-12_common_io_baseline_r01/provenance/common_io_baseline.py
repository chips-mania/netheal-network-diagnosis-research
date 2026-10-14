"""Local, resumable 0-update diagnosis baseline; no network/API calls."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import argparse
import collections
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from netheal_research.common_diagnosis_io import (
    prepare, combine, diagnosis_options, digest, public_audit, MAX_LEN, HEAD_LEN, LEGACY)
from jev_context_codec_v4 import dumps

RAW = ROOT / 'netheal-topology-generalization/results/laya_official_20261012_v1'
BASE = ROOT / '.runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148'
DEFAULT_OUT = ROOT / 'runs/official_benchmark/laya/2026-10-12_common_io_baseline_r01'
SPEC = ROOT / 'docs/specs/laya_common_io_baseline_20261012.md'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def core(x):
    if isinstance(x, dict):
        return {k: core(v) for k, v in x.items() if k not in {'timestamp', 'step_count', 'episode_progress', 'episode_metadata', 'reward', 'reward_breakdown', 'recent_diagnostics'}}
    if isinstance(x, list):
        return [core(v) for v in x]
    return x


def build_case(c, schemas):
    # Only these public fields are transferred. Never transfer case questions or labels.
    history = []
    for event in c['trace']:
        if event['tool'] == 'submit_diagnosis':
            break
        history.append(event)
    assert history and history[0]['tool'] == 'scan_network'
    task = ('Diagnose the network fault. The simulated network has an injected fault '
            '(device failure, link failure, misconfiguration, or performance degradation). '
            'Use the diagnostic tools to explore the network, identify the fault, '
            'and submit your diagnosis using the submit_diagnosis tool.')
    objective = ('Identify the fault type and location within 100 tool calls. '
                 'Call submit_diagnosis(fault_type, location) with your answer. '
                 'Each tool call (except list_actions) consumes one step from your budget.')
    state = dict(task_description=task, objective=objective, hint=c['hint'],
                 network_size=history[0]['result']['info']['network_size'], max_steps=100,
                 tool_schemas=schemas, history=history, solver_turn=len(history))
    public_audit(state)
    return state


def same_location(a, b):
    return a == b or ('->' in a and '->' in b and set(a.split('->')) == set(b.split('->')))


def summarize(rows):
    result = {}
    for split in ('train', 'development'):
        rs = [r for r in rows if r['split'] == split]
        if not rs:
            continue
        f1 = []
        for label in ('device_failure', 'link_failure', 'misconfiguration', 'performance_degradation'):
            tp = sum(r['truth']['type'] == label and r['predicted_type'] == label for r in rs)
            fp = sum(r['truth']['type'] != label and r['predicted_type'] == label for r in rs)
            fn = sum(r['truth']['type'] == label and r['predicted_type'] != label for r in rs)
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        keys = ('type_correct', 'location_correct', 'joint_correct', 'oracle_location_correct', 'type_order_changed', 'location_order_changed')
        result[split] = {'cases': len(rs), **{k: sum(r[k] for r in rs) / len(rs) for k in keys},
                         'type_macro_f1': sum(f1) / 4,
                         'confusion': {t: dict(collections.Counter(r['predicted_type'] for r in rs if r['truth']['type'] == t)) for t in sorted({r['truth']['type'] for r in rs})}}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--audit-only', action='store_true')
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for d in ('provenance', 'logs', 'metrics', 'inputs', 'predictions'):
        (out / d).mkdir(exist_ok=True)
    import torch
    import psutil
    import laya.train as training
    import laya.common as common
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    torch.manual_seed(1012)
    config = {'model': str(BASE.relative_to(ROOT)), 'updates': 0, 'dtype': 'float32',
              'max_len': MAX_LEN, 'head_max_len': HEAD_LEN, 'temperature': 1.,
              'seed': 1012, 'cases': 60, 'split': {'train': 40, 'development': 20},
              'order_control': 'reverse', 'training': False, 'network_calls': False}
    sources = [Path(__file__), ROOT / 'src/netheal_research/common_diagnosis_io.py', SPEC,
               LEGACY / 'jev_context_codec_v4.py', LEGACY / 'jev_tree_partition_v4.py',
               Path(training.__file__), Path(common.__file__), BASE / 'model.safetensors', BASE / 'rl_agent_config.json']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    if (out / 'config.json').exists():
        assert read(out / 'config.json') == config, 'Resume config changed'
        assert read(out / 'provenance/source_hashes.json') == hashes, 'Resume source changed'
    else:
        save(out / 'config.json', config)
        save(out / 'provenance/source_hashes.json', hashes)
        for p in sources:
            if p.suffix in ('.py', '.md'):
                shutil.copyfile(p, out / 'provenance' / p.name)
    paths = sorted((RAW / 'cases').glob('*.json'))
    cases = [read(p) for p in paths]
    assert len(cases) == 60
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    input_hashes[str((RAW / 'tool_schemas.json').relative_to(ROOT))] = sha(RAW / 'tool_schemas.json')
    if (out / 'provenance/raw_hashes.json').exists():
        assert read(out / 'provenance/raw_hashes.json') == input_hashes
    save(out / 'provenance/raw_hashes.json', input_hashes)
    schemas = read(RAW / 'tool_schemas.json')
    states = [build_case(c, schemas) for c in cases]
    audit_rows = []
    for c, state in zip(cases, states):
        hist = state['history']
        counts = collections.Counter(e['tool'] for e in hist)
        types = diagnosis_options(c['catalog'])
        assert c['truth']['type'] in types
        assert c['truth']['location'] in diagnosis_options(c['catalog'], c['truth']['type'])
        assert all(len(diagnosis_options(c['catalog'], ft)) <= 255 for ft in types)
        n = state['network_size']
        audit_rows.append({'seed': c['seed'], 'split': c['split'], 'snapshot_id': c['snapshot_id'],
             'fingerprint': c['fingerprint'], 'public_core_hash': digest(core(hist)),
             'public_state_hash': digest(state), 'observed_steps': len(hist), 'tool_counts': counts,
             'actual_devices': n, 'all_status_and_interfaces': counts['check_status'] == counts['check_interfaces'] == n,
             'observation_sufficiency': 'not_counterfactually_validated', 'excluded': False})
    for field in ('seed', 'snapshot_id', 'fingerprint', 'public_core_hash'):
        assert len({r[field] for r in audit_rows}) == len(audit_rows), 'Duplicate ' + field
    # Near duplicate audit: Jaccard overlap of normalized public action-result facts across splits.
    facts = [{digest(core(e['result']['info'].get('action_result'))) for e in s['history']} for s in states]
    near = []
    for i, c in enumerate(cases):
        for j in range(i):
            if c['split'] != cases[j]['split']:
                score = len(facts[i] & facts[j]) / len(facts[i] | facts[j])
                if score >= .8:
                    near.append({'seeds': [c['seed'], cases[j]['seed']], 'jaccard': score})
    save(out / 'data_audit.json', {'cases': audit_rows, 'cross_split_near_duplicates_threshold': .8,
         'cross_split_near_duplicates': near, 'all_cases_included': True, 'targets_present_in_candidates': True,
         'historical_official_overlap_audit': read(RAW / 'data_audit.json'),
         'limitation': 'No counterfactual identifiability proof; no success-based selection.'})
    if args.audit_only:
        print('AUDIT PASSED', flush=True)
        return
    started = time.perf_counter()
    save(out / 'manifest.json', {'status': 'running', 'config': 'config.json', 'completed_cases': len(list((out / 'metrics').glob('case_*.json')))})
    model, tok, cfg = training.load_checkpoint(str(BASE))
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.88)
    model.to('cuda').eval()
    torch.cuda.reset_peak_memory_stats()
    forward_count = 0
    tokens_count = 0

    def choose(c, state, purpose, options):
        nonlocal forward_count, tokens_count
        key = digest({'state': digest(state), 'purpose': purpose, 'options': options})
        inp = out / 'inputs' / f'{key}.json'
        pred = out / 'predictions' / f'{key}.json'
        if pred.exists():
            return read(pred)['selected']
        if inp.exists():
            plan = read(inp)
        else:
            plan = prepare(tok, state, purpose, options)
            save(inp, plan)
        probs = []
        t = time.perf_counter()
        for chunk in plan['chunks']:
            q = training.to_internal('decision', chunk['input']['questions']['decision'])
            ids, markers = common.build_sequence(tok, chunk['input']['state'], q, MAX_LEN, HEAD_LEN)
            encoded = {'ids': ids, 'markers': markers, 'qtype': common.QTYPES['choice'], 'target': [0.] * len(options)}
            batch = common.collate_items([[encoded]], tok.pad_token_id)
            with torch.inference_mode():
                logits = training._forward(model, batch, torch.device('cuda'), False, False)
                p = torch.softmax(logits[0, :len(options)], dim=-1).cpu().tolist()
            assert len(p) == len(options) and all(0 <= v <= 1 for v in p) and abs(sum(p) - 1) < 1e-5
            probs.append({f'c{i:03d}': v for i, v in enumerate(p)})
            forward_count += 1
            tokens_count += len(ids)
        response = combine(plan, probs)
        selected = options[int(response['answers']['decision']['choice'][1:])]
        save(pred, {'case_id': c['seed'], 'input_hash': sha(inp), 'selected': selected, 'response': response,
                    'chunk_probabilities': probs, 'forwards': len(probs), 'tokens': sum(x['tokens'] for x in plan['chunks']),
                    'seconds': time.perf_counter() - t, 'temperature': 1.})
        print(json.dumps({'seed': c['seed'], 'purpose': purpose, 'options': len(options), 'chunks': len(probs), 'seconds': round(time.perf_counter() - t, 2)}), flush=True)
        return selected

    try:
        for c, state in zip(cases, states):
            dest = out / 'metrics' / f'case_{c["seed"]}.json'
            if dest.exists():
                continue
            types = diagnosis_options(c['catalog'])
            fpurpose = 'Select fault_type for submit_diagnosis.'
            ft = choose(c, state, fpurpose, types)
            locs = diagnosis_options(c['catalog'], ft)
            lpurpose = 'Select location for predicted fault_type ' + ft + '.'
            loc = choose(c, state, lpurpose, locs)
            reverse_ft = choose(c, state, fpurpose, list(reversed(types)))
            reverse_loc = choose(c, state, lpurpose, list(reversed(locs)))
            true_ft = c['truth']['type']
            oracle = choose(c, state, 'Select location for predicted fault_type ' + true_ft + '.', diagnosis_options(c['catalog'], true_ft))
            tc = ft == true_ft
            lc = same_location(loc, c['truth']['location'])
            save(dest, {'case_id': c['seed'], 'split': c['split'], 'truth': c['truth'], 'predicted_type': ft,
                        'predicted_location': loc, 'oracle_type_location': oracle, 'type_correct': tc,
                        'location_correct': lc, 'joint_correct': tc and lc,
                        'oracle_location_correct': same_location(oracle, c['truth']['location']),
                        'type_order_changed': ft != reverse_ft, 'location_order_changed': not same_location(loc, reverse_loc)})
            rows = [read(p) for p in sorted((out / 'metrics').glob('case_*.json'))]
            save(out / 'metrics/summary.json', summarize(rows))
            save(out / 'manifest.json', {'status': 'running', 'completed_cases': len(rows), 'total_cases': 60})
            print('CASE COMPLETE', c['seed'], len(rows), flush=True)
        predictions = [read(p) for p in (out / 'predictions').glob('*.json')]
        save(out / 'resources.json', {'invocation_seconds': time.perf_counter() - started,
             'questions': len(predictions), 'forward_calls': sum(p['forwards'] for p in predictions),
             'tokens': sum(p['tokens'] for p in predictions), 'gpu_peak_allocated': torch.cuda.max_memory_allocated(),
             'gpu_peak_reserved': torch.cuda.max_memory_reserved(), 'ram_rss_end': psutil.Process().memory_info().rss,
             'torch_version': torch.__version__, 'python': sys.version, 'gpu': torch.cuda.get_device_name(),
             'disk_free': shutil.disk_usage(out).free})
        assert len(list((out / 'metrics').glob('case_*.json'))) == 60
        save(out / 'manifest.json', {'status': 'completed', 'completed_cases': 60, 'updates': 0, 'api_calls': 0})
        print('BASELINE COMPLETE', flush=True)
    except Exception as exc:
        save(out / 'manifest.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
             'completed_cases': len(list((out / 'metrics').glob('case_*.json')))})
        raise


if __name__ == '__main__':
    main()
