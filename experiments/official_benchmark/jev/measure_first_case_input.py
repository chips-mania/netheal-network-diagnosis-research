"""Paid, bounded paired replay of one case; never executes environment actions."""
import json
import time
import hashlib
import statistics
from pathlib import Path
import httpx
from dotenv import dotenv_values
from compare_first_case_input import ROOT, OUT, guard, save


def partitions(body):
    if guard(body) <= 24000:
        return [(body, 1)]
    state = json.loads(body['state'])
    history = state.pop('history')
    def wrap(events, first):
        s = dict(state, history=events, history_partition={'first': first, 'count': len(events), 'total': len(history),
            'rule': 'This is a contiguous evidence partition. Other partitions are evaluated separately with the same options.'})
        return dict(body, state=json.dumps(s, ensure_ascii=False, separators=(',', ':')))
    result, current, first = [], [], 0
    for e in history:
        if guard(wrap(current + [e], first)) <= 24000:
            current.append(e)
        else:
            assert current, 'Single event too large'
            result.append((wrap(current, first), len(current)))
            first += len(current)
            current = [e]
            assert guard(wrap(current, first)) <= 24000
    if current:
        result.append((wrap(current, first), len(current)))
    assert [e for b, _ in result for e in json.loads(b['state'])['history']] == history
    return result


def main():
    prepared = []
    for folder in sorted((OUT / 'requests').iterdir()):
        for variant in (('old', 'new') if int(folder.name) % 2 else ('new', 'old')):
            body = json.loads((folder / f'{variant}_request.json').read_text(encoding='utf-8'))
            parts = [(body, 1)] if variant == 'old' else partitions(body)
            prepared.append((folder, variant, parts))
    plan = {'mode': 'fixed historical observations; not a fresh end-to-end rollout', 'case_count': 1,
            'judgments': 22, 'calls': sum(len(p) for _, _, p in prepared), 'temperature': 'API defaults unchanged',
            'order': 'alternate old/new by judgment', 'retries': 0,
            'aggregation': 'history-event weighted probabilities; single request provider choice',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(OUT / 'measurement_protocol.json', plan)
    key = dotenv_values(ROOT / '.env').get('TYPESAFE_API_KEY')
    assert key, 'API key missing'
    print('Planned requests', plan['calls'], flush=True)
    with httpx.Client(headers={'Authorization': 'Bearer ' + key}, timeout=120, follow_redirects=False) as client:
        for folder, variant, parts in prepared:
            dest = folder / f'{variant}_measured.json'
            if dest.exists():
                continue
            responses = []
            for i, (body, weight) in enumerate(parts):
                rawpath = folder / f'{variant}_part_{i:02d}.json'
                if rawpath.exists():
                    record = json.loads(rawpath.read_text(encoding='utf-8'))
                    assert record['request'] == body and record['status'] == 200
                else:
                    time.sleep(.5)
                    start = time.perf_counter()
                    response = client.post('https://api.typesafe.ai/v1/systemone', json=body)
                    elapsed = time.perf_counter() - start
                    record = {'request': body, 'weight': weight, 'seconds': elapsed, 'status': response.status_code}
                    if response.status_code == 200:
                        data = response.json()
                        record['response'] = {k: data[k] for k in ('answers', 'usage', 'model', 'id') if k in data}
                    save(rawpath, record)
                    assert response.status_code == 200, f'HTTP {response.status_code}; no retry'
                ans = record['response']['answers']['decision']
                criteria = body['questions']['decision']['criteria']
                assert ans['choice'] in criteria and set(ans['probabilities']) == set(criteria)
                assert all(0 <= p <= 1 for p in ans['probabilities'].values())
                responses.append(record)
            keys = list(criteria)
            scores = {k: sum(r['weight'] * r['response']['answers']['decision']['probabilities'][k] for r in responses) / sum(r['weight'] for r in responses) for k in keys}
            choice = responses[0]['response']['answers']['decision']['choice'] if len(responses) == 1 else max(keys, key=scores.get)
            save(dest, {'choice': choice, 'selected': criteria[choice], 'probabilities': scores,
                       'calls': len(responses), 'http_seconds': sum(r['seconds'] for r in responses),
                       'input_tokens': sum(r['response'].get('usage', {}).get('input_tokens', 0) for r in responses)})
            print(folder.name, variant, len(responses), criteria[choice], flush=True)
    rows = []
    for folder in sorted((OUT / 'requests').iterdir()):
        old = json.loads((folder / 'old_measured.json').read_text(encoding='utf-8'))
        new = json.loads((folder / 'new_measured.json').read_text(encoding='utf-8'))
        historical = json.loads((folder / 'old_response.json').read_text(encoding='utf-8'))['answers']['decision']['choice']
        rows.append({'judgment': int(folder.name), 'old': old, 'new': new, 'same_choice': old['choice'] == new['choice'], 'old_matches_historical': old['choice'] == historical})
    totals = {v: {'calls': sum(r[v]['calls'] for r in rows), 'http_seconds': sum(r[v]['http_seconds'] for r in rows),
                  'median_judgment_http_seconds': statistics.median(r[v]['http_seconds'] for r in rows),
                  'input_tokens': sum(r[v]['input_tokens'] for r in rows)} for v in ('old', 'new')}
    save(OUT / 'measurement.json', {'status': 'completed', 'totals': totals, 'same_choices': sum(r['same_choice'] for r in rows),
         'old_matches_historical': sum(r['old_matches_historical'] for r in rows), 'rows': rows,
         'limitation': 'Single replay per variant. New choices do not alter subsequent historical observations; no new episode success rate.'})
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
