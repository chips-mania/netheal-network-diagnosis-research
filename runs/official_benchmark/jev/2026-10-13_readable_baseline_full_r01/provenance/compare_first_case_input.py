"""Inspect a readable rendering of the first official case, without altering evidence."""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'netheal-topology-generalization/scripts'))
from jev_context_codec_v4 import decode

SID = '00bd290a-df64-46a6-8f06-f087bde022a9'
SOURCE = ROOT / 'runs/official_benchmark/jev/netheal_official_jev_20261011_v4/episodes' / SID / 'attempt_00/requests.jsonl'
OUT = ROOT / 'runs/official_benchmark/jev/2026-10-13_first_case_readable_input_r01'


def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')


def simplify_schema(s):
    # Remove annotation titles only; property names, types, required and constraints survive.
    if isinstance(s, list):
        return [simplify_schema(v) for v in s]
    if not isinstance(s, dict):
        return s
    return {k: simplify_schema(v) for k, v in s.items() if k != 'title'}


def readable(state):
    result = copy.deepcopy(state)
    result['tool_schemas'] = [{'name': t['name'], 'description': t['description'],
                               'inputSchema': simplify_schema(t['inputSchema'])}
                              for t in state['tool_schemas']]
    assert result['history'] == state['history']
    assert {k:v for k,v in result.items() if k != 'tool_schemas'} == {k:v for k,v in state.items() if k != 'tool_schemas'}
    return result


def guard(body):
    return len(body['state'].encode()) + len(json.dumps(body['questions'], ensure_ascii=False).encode()) + 512


def main():
    records = [json.loads(x) for x in SOURCE.read_text(encoding='utf-8').splitlines()]
    assert len(records) == 22
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, record in enumerate(records, 1):
        old = record['request']
        state = decode(json.loads(old['state']))
        assert 'evidence_fragments' not in state and 'history_partition' not in state
        clean = readable(state)
        new = dict(old, state=json.dumps(clean, ensure_ascii=False, separators=(',', ':')))
        assert old['questions'] == new['questions']
        assert json.loads(new['state'])['history'] == state['history']
        folder = OUT / 'requests' / f'{i:02d}'
        save(folder / 'old_request.json', old)
        save(folder / 'new_request.json', new)
        save(folder / 'readable_state.json', clean)
        save(folder / 'old_response.json', record['response'])
        rows.append({'request': i, 'solver_turn': state['solver_turn'], 'history_events': len(state['history']),
            'question': old['questions']['decision']['instructions'], 'options': len(old['questions']['decision']['criteria']),
            'old_state_bytes': len(old['state'].encode()), 'new_state_bytes': len(new['state'].encode()),
            'old_guard_bytes': guard(old), 'new_guard_bytes': guard(new), 'new_fits_existing_guard': guard(new) <= 24000,
            'old_choice': record['response']['answers']['decision']['choice'],
            'old_selected': old['questions']['decision']['criteria'][record['response']['answers']['decision']['choice']],
            'old_seconds': record['seconds'], 'history_equal': True, 'questions_equal': True})
    save(OUT / 'comparison.json', {'snapshot_id': SID, 'mode': 'frozen_observation_input_comparison',
         'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
         'changed': ['decode compression', 'keep only tool name/description/inputSchema', 'remove schema titles'],
         'unchanged': ['all observations and history', 'task/objective/hint/budget', 'questions and candidate order'],
         'requests': rows, 'new_requests_over_guard': sum(not r['new_fits_existing_guard'] for r in rows), 'api_calls': 0})
    print(json.dumps({'requests': len(rows), 'first': rows[0], 'last': rows[-1],
          'over_guard': [r['request'] for r in rows if not r['new_fits_existing_guard']],
          'old_total_state_bytes': sum(r['old_state_bytes'] for r in rows),
          'new_total_state_bytes': sum(r['new_state_bytes'] for r in rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
