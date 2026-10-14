"""Verify every live model input against only its preceding raw tool results."""
import hashlib
import json
from pathlib import Path
from structured_state import StructuredState
from verify_structured_state import audit

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'runs/official_benchmark/jev/2026-10-13_structured_state_r01'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def main():
    protocol = read(OUT / 'protocol.json')
    for name, expected in protocol['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    results = sorted((OUT / 'episodes').glob('*/result.json'))
    assert len(results) == 100
    count = measurements = 0
    for path in results:
        folder = path.parent
        events = lines(folder / 'tools.jsonl')
        measurements += audit(events)
        requests = lines(folder / 'request_started.jsonl')
        assert len(requests) == read(path)['model_requests']
        for record in requests:
            state = json.loads(record['request']['state'])
            n = sum(a['count'] for a in state['diagnostic_evidence_and_action_memory'])
            assert n <= len(events)
            for action in state['diagnostic_evidence_and_action_memory']:
                for version in action['measurements']:
                    for point in version['observed_at']:
                        if point['timestamp'] is not None:
                            assert point['timestamp'] <= record['unix']
            # Notices do not consume environment actions. These runs should not
            # contain missing-candidate notices; fail loudly if the assumption changes.
            assert not state['adapter_notices']
            raw = {'history': events[:n], 'objective': state['task']['objective'],
                   'hint': state['task']['hint'], 'network_size': state['task']['network_size'],
                   'max_steps': 100, 'solver_turn': state['task']['solver_turn']}
            expected = StructuredState().render(raw)
            assert expected == {k: v for k, v in state.items()
                                if k not in ('tool_schemas', 'original_agent_system_prompt')}
            assert 'history' not in state
            count += 1
        assert not (folder / 'partition_decisions.jsonl').exists()
    result = {'status': 'passed', 'episodes': len(results), 'requests_verified': count,
              'measurements_verified': measurements, 'source_hashes_verified': True,
              'every_input_equals_raw_observation_prefix_reduction': True,
              'no_partition_decisions': True}
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
