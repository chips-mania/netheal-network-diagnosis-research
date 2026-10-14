"""Single live episode, original decisions, exclusive wall-time accounting, no fixed sleep."""
import asyncio
import collections
import contextlib
import functools
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'netheal-topology-generalization/scripts/jev_official_eval_v4.py'
OUT = ROOT / 'runs/official_benchmark/jev/2026-10-13_first_case_latency_r01'
SID = '00bd290a-df64-46a6-8f06-f087bde022a9'
totals = collections.defaultdict(float)
counts = collections.Counter()
stack = []
last = 0.0


@contextlib.contextmanager
def section(name):
    global last
    now = time.perf_counter()
    if stack:
        totals[stack[-1]] += now - last
    stack.append(name)
    counts[name] += 1
    last = now
    try:
        yield
    finally:
        now = time.perf_counter()
        totals[stack.pop()] += now - last
        last = now


def sync(fn, name):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with section(name):
            return fn(*args, **kwargs)
    return wrapped


def asynchronous(fn, name):
    @functools.wraps(fn)
    async def wrapped(*args, **kwargs):
        with section(name):
            return await fn(*args, **kwargs)
    return wrapped


async def main():
    OUT.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(SOURCE.parent))
    source = SOURCE.read_text(encoding='utf-8')
    before = 'await asyncio.sleep(max(0,0.5-(time.monotonic()-LAST_REQUEST)))'
    assert source.count(before) == 1
    modified = source.replace(before, 'await asyncio.sleep(0)')
    module = types.ModuleType('profiled_jev')
    module.__file__ = str(SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(modified, str(SOURCE), 'exec'), module.__dict__)
    shutil.copy2(module.OUT / 'tool_schemas.json', OUT / 'tool_schemas.json')
    module.OUT = OUT
    module.save(OUT / 'protocol.json', {
        'case': SID, 'concurrency': 1, 'fixed_sleep_seconds': 0,
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'change': 'Only fixed sleep replaced with asyncio.sleep(0); timing wrappers added.',
        'retry': False, 'max_turns': 100, 'deadline_seconds': 1200,
        'timing': 'Exclusive nested wall time; imports/snapshot loading excluded; actual tools executed.',
        'official_limits_checked': 'https://docs.typesafe.ai/models: 80 requests/s, 100K tokens/s on 2026-10-13',
    })
    (OUT / 'executed_eval.py').write_text(modified, encoding='utf-8')
    for name, category in [('encode_checked', 'compression_roundtrip'),
                           ('partition_state', 'partition_other'), ('runtime', 'environment_setup'),
                           ('append', 'log_write'), ('save', 'log_write')]:
        setattr(module, name, sync(getattr(module, name), category))
    original_call = module.call
    async def call(server, name, args):
        category = 'candidate_list_actions' if name == 'list_actions' else 'environment_tools'
        with section(category):
            return await original_call(server, name, args)
    module.call = call
    original_init = module.Jev.__init__
    def init(self, *args, **kwargs):
        with section('http_client_setup'):
            original_init(self, *args, **kwargs)
        self.client.post = asynchronous(self.client.post, 'api_http')
    module.Jev.__init__ = init
    module.Jev.choose = asynchronous(module.Jev.choose, 'decision_adapter_other')
    snapshot = next(s for s in module.snapshots() if s['snapshot_id'] == SID)
    key = module.dotenv_values(ROOT / '.env').get('TYPESAFE_API_KEY')
    assert key, 'API key missing'
    start = time.perf_counter()
    with section('episode_other'):
        status = await module.episode(snapshot, key, OUT / 'episode')
    elapsed = time.perf_counter() - start
    result = json.loads((OUT / 'episode/result.json').read_text(encoding='utf-8'))
    requests = [json.loads(line) for line in (OUT / 'episode/requests.jsonl').read_text(encoding='utf-8').splitlines()]
    report = {'status': status, 'total_seconds': elapsed, 'accounted_seconds': sum(totals.values()),
              'fixed_wait_seconds': 0, 'categories': {k: {'seconds': v, 'percent': 100*v/elapsed,
              'count': counts[k]} for k, v in totals.items()},
              'calls': len(requests), 'input_tokens': sum(r.get('response', {}).get('usage', {}).get('input_tokens', 0) for r in requests),
              'result': result, 'limitation': 'One autonomous rollout; choices may differ from historical run. No retries.'}
    (OUT / 'profile.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
