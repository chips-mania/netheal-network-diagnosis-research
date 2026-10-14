"""Profile one real LAYA episode with unchanged official baseline settings."""
import asyncio
import importlib.util
import json
from pathlib import Path
import shutil
import time
import official_baseline as baseline

ROOT = baseline.ROOT
OUT = ROOT / 'runs/official_benchmark/laya/2026-10-13_first_case_latency_r01'
SID = '00bd290a-df64-46a6-8f06-f087bde022a9'
spec = importlib.util.spec_from_file_location('timing_helpers', ROOT / 'experiments/official_benchmark/jev/profile_first_case.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


async def main():
    OUT.mkdir(parents=True, exist_ok=False)
    env = baseline.env
    shutil.copy2(env.OUT / 'tool_schemas.json', OUT / 'tool_schemas.json')
    env.OUT = OUT
    env.save(OUT / 'protocol.json', {'case': SID, 'max_turns': 100, 'deadline_seconds': 1200,
        'concurrency': 1, 'fixed_wait_seconds': 0, 'dtype': 'float32', 'seed': 1012,
        'max_len': baseline.MAX_LEN, 'head_len': baseline.HEAD_LEN,
        'checkpoint': str(baseline.BASE), 'baseline_sha256': baseline.sha(Path(baseline.__file__)),
        'timing': 'Exclusive wall time. Model load separate; Python imports/snapshot loading excluded. Forward includes collation, transfer, softmax and CPU result synchronization.',
        'change': 'Timing wrappers only; unchanged input, model, temperature, partitioning and environment loop.'})
    snapshot = next(s for s in env.snapshots() if s['snapshot_id'] == SID)
    start = time.perf_counter()
    backend = baseline.Backend()
    backend.torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    print(json.dumps({'model_load_seconds': load_seconds}), flush=True)
    baseline.Laya.backend = backend
    env.Jev = baseline.Laya
    for module, name, category in [
        (env, 'runtime', 'environment_setup'), (env, 'append', 'log_write'),
        (env, 'save', 'log_write'), (env, 'encode_checked', 'compression_roundtrip'),
        (baseline, 'encode_checked', 'compression_roundtrip'),
        (env, 'partition_state', 'partition_other'),
        (baseline.Backend, 'partitions', 'partition_other'),
        (baseline.Backend, 'encode', 'tokenization_checks'),
        (baseline.Backend, 'forward', 'model_forward')]:
        setattr(module, name, timing.sync(getattr(module, name), category))
    original_call = env.call
    async def call(server, name, args):
        with timing.section('candidate_list_actions' if name == 'list_actions' else 'environment_tools'):
            return await original_call(server, name, args)
    env.call = call
    baseline.Laya.choose = timing.asynchronous(baseline.Laya.choose, 'decision_adapter_other')
    start = time.perf_counter()
    with timing.section('episode_other'):
        status = await env.episode(snapshot, None, OUT / 'episode')
    elapsed = time.perf_counter() - start
    result = baseline.read(OUT / 'episode/result.json')
    report = {'status': status, 'total_seconds': elapsed, 'model_load_seconds': load_seconds,
        'accounted_seconds': sum(timing.totals.values()), 'fixed_wait_seconds': 0,
        'categories': {k: {'seconds': v, 'percent': 100*v/elapsed, 'count': timing.counts[k]}
                       for k, v in timing.totals.items()},
        'result': result, 'gpu': backend.torch.cuda.get_device_name(),
        'gpu_peak_allocated': backend.torch.cuda.max_memory_allocated(),
        'gpu_peak_reserved': backend.torch.cuda.max_memory_reserved(),
        'limitation': 'One autonomous rollout; same case but observations and decisions can differ from Jev.'}
    (OUT / 'profile.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
