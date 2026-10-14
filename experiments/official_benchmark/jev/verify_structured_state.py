"""Offline causal replay audit; never uses evaluator labels as reducer inputs."""
import copy
import json
from pathlib import Path
from structured_state import StructuredState, dumps, measurement

ROOT = Path(__file__).resolve().parents[3]


def audit(events):
    reducer = StructuredState()
    raw = {'history': [], 'objective': 'diagnose', 'network_size': 9, 'max_steps': 100}
    for event in events:
        raw['history'].append(event)
        output = reducer.render(raw)
        if 'result' in event:
            assert output['latest_observation']['observation'] == measurement(event)
            assert output['latest_observation']['arguments'] == event['arguments']
    # Reconstruct all retained measurements in their original event order.
    rebuilt = {}
    for action in reducer.memory.values():
        for version in action['measurements']:
            for point in version['observed_at']:
                value = copy.deepcopy(version['result'])
                original = measurement(events[point['event']-1])
                if 'timestamp' in original:
                    value['timestamp'] = point['timestamp']
                assert value == original
                rebuilt[point['event']] = value
        expected = [measurement(e) for e in events if e.get('tool') == action['tool']
                    and e.get('arguments') == action['arguments'] and 'result' in e]
        assert action['count'] == len(expected)
        comparable = [{k: v for k, v in x.items() if k != 'timestamp'} for x in expected]
        assert action['result_changed'] == any(x != comparable[0] for x in comparable[1:])
    assert len(rebuilt) == sum('result' in e for e in events)
    # Prefix rendering cannot depend on future events or hidden fields.
    midpoint = len(events)//2
    prefix = dict(raw, history=events[:midpoint])
    first = StructuredState().render(prefix)
    poison = dict(prefix, ground_truth='FUTURE_SECRET', reward_breakdown={'secret': 999})
    assert first == StructuredState().render(poison)
    frozen = copy.deepcopy(first)
    another = StructuredState()
    old = another.render(prefix)
    another.render(raw)
    assert old == frozen
    return len(rebuilt)


def main():
    def event(status, timestamp, step):
        return {'tool': 'ping', 'arguments': {'source': 'a', 'destination': 'b'},
                'result': {'info': {'step_count': step, 'reward_breakdown': {'secret': 999},
                    'action_result': {'source': 'a', 'destination': 'b', 'timestamp': timestamp,
                        'result': {'success': True, 'data': {'latency_ms': status, 'packet_loss': 0.25,
                            'hops': [{'node': 'b', 'latency_ms': status}]}}}}}}
    synthetic = [event(5, 1, 1), event(5, 2, 2), event(50, 3, 3), event(5, 4, 4)]
    audit(synthetic)
    root = ROOT / 'runs/official_benchmark/jev/2026-10-13_readable_baseline_full_r01/episodes'
    paths = sorted(root.glob('*/tools.jsonl'))
    assert len(paths) == 100
    total = 0
    for p in paths:
        events = [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
        total += audit(events)
    report = {'status': 'passed', 'episodes': len(paths), 'measurements_verified': total,
              'checks': ['all numeric measurement fields and timestamps roundtrip',
                         'action counts and result_changed compare all observations',
                         'latency changes and return to prior value retained',
                         'prefix-only rendering and hidden-field independence',
                         'returned state cannot mutate after future updates'],
              'strict_lossless_claim': False}
    dest = ROOT / 'runs/official_benchmark/jev/2026-10-13_structured_state_validation.json'
    dest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(dumps(report))


if __name__ == '__main__':
    main()
