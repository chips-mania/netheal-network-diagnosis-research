"""Exercise the actual frozen episode loop using a scripted public-choice client."""
import asyncio
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('official_laya', ROOT / 'experiments/official_benchmark/laya/official_baseline.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_frozen_loop_predicted_type_and_public_candidates(tmp_path, monkeypatch):
    env = runner.env
    s = runner.read(env.OUT / 'development_snapshot.json')
    seen = []
    class Scripted:
        def __init__(self, key, folder, deadline):
            self.calls = 0
            self.client = self
        async def aclose(self):
            pass
        async def choose(self, state, purpose, options):
            runner.audit(state)
            self.calls += 1
            seen.append((purpose, options))
            if purpose == 'Select the next tool.':
                return 'submit_diagnosis'
            if purpose == 'Select fault_type for submit_diagnosis.':
                return 'link_failure'
            assert purpose == 'Select location for predicted fault_type link_failure.'
            assert all('->' in x for x in options)
            return options[0]
    monkeypatch.setattr(env, 'Jev', Scripted)
    status = asyncio.run(env.episode(s, None, tmp_path / 'episode'))
    assert status == 'completed'
    assert len(seen) == 3
    result = runner.read(tmp_path / 'episode/result.json')
    assert result['metrics']['predicted_type'] == 'link_failure'
    assert result['model_requests'] == 3
    events = [json.loads(x) for x in (tmp_path / 'episode/tools.jsonl').read_text().splitlines()]
    assert len(events) == 1 and events[0]['arguments']['fault_type'] == 'link_failure'


def test_timeout_never_executes_tool(tmp_path, monkeypatch):
    env = runner.env
    s = runner.read(env.OUT / 'development_snapshot.json')
    class Timeout:
        def __init__(self, key, folder, deadline):
            self.calls = 0
            self.client = self
        async def aclose(self):
            pass
        async def choose(self, *args):
            raise TimeoutError('test deadline')
    monkeypatch.setattr(env, 'Jev', Timeout)
    assert asyncio.run(env.episode(s, None, tmp_path / 'episode')) == 'error_TimeoutError'
    assert not (tmp_path / 'episode/tools.jsonl').exists()
    result = runner.read(tmp_path / 'episode/result.json')
    assert not result['metrics']['diagnosis_success']


def test_model_probabilities_are_weighted_with_stable_argmax(tmp_path):
    class Fake:
        def partitions(self, *args):
            q = runner.env.question('choose', ['a', 'b'])
            return q, [{}, {}], [1, 3], [(True, '{}', {'ids': [1]}), (True, '{}', {'ids': [2]})], 'test'
        def forward(self, encoded, count):
            return ([1., 0.] if encoded['ids'] == [1] else [0., 1.]), 1.
    client = runner.Laya(None, tmp_path, runner.time.monotonic() + 10)
    client.backend = Fake()
    selected = asyncio.run(client.choose({'solver_turn': 0}, 'choose', ['a', 'b']))
    assert selected == 'b' and client.calls == 2 and client.questions == 1
