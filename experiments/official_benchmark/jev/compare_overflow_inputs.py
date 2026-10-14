"""Replay the 39 historical overflow requests under three fixed representations."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import time
import httpx
from dotenv import dotenv_values
from compare_first_case_input import ROOT, readable, save
from jev_context_codec_v4 import encode_checked

OUT = ROOT / 'runs/official_benchmark/jev/2026-10-13_overflow_input_comparison_r01'
SOURCE = ROOT / 'runs/official_benchmark/jev/netheal_official_jev_20261011_v2/episodes'


def main():
    cases = []
    for path in sorted(SOURCE.glob('*/attempt_00/requests.jsonl')):
        for index, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            old = json.loads(line)
            if old['status'] != 400:
                continue
            assert old['error_type'] == 'max_tokens_exceeded'
            body = old['request']
            state = body['state']
            assert isinstance(state, dict)
            variants = {'original': body,
                        'compressed': dict(body, state=encode_checked(state)),
                        'readable': dict(body, state=json.dumps(readable(state), ensure_ascii=False, separators=(',', ':')))}
            assert all(b['questions'] == body['questions'] for b in variants.values())
            cases.append((path.parents[1].name, index, variants))
    assert len(cases) == 39
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'protocol.json', {'cases': 39, 'max_calls': 117, 'model': 'jev-1.13.0',
        'variants': ['original v2 body, state object unchanged', 'v4 lossless compression, state string',
                     'prior readable transformation: keep all history, simplify tool schemas, state string'],
        'questions_and_options': 'historical exact', 'guard': None, 'partition': None,
        'interval': 'at least one second between starts; sequential', 'retry': 0,
        'purpose': 'API acceptance and reported token usage; not autonomous episodes or accuracy evaluation',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    key = dotenv_values(ROOT / '.env').get('TYPESAFE_API_KEY')
    assert key
    rows = []
    last_start = 0.0
    with httpx.Client(headers={'Authorization': 'Bearer ' + key}, timeout=120, follow_redirects=False) as client:
        for n, (sid, index, variants) in enumerate(cases):
            names = list(variants)
            names = names[n % 3:] + names[:n % 3]
            row = {'snapshot_id': sid, 'source_line': index, 'variants': {}}
            for name in names:
                body = variants[name]
                dest = OUT / 'requests' / sid / (name + '.json')
                if dest.exists():
                    record = json.loads(dest.read_text(encoding='utf-8'))
                    assert record['request'] == body
                else:
                    time.sleep(max(0, 1 - (time.perf_counter() - last_start)))
                    last_start = time.perf_counter()
                    response = client.post('https://api.typesafe.ai/v1/systemone', json=body)
                    seconds = time.perf_counter() - last_start
                    record = {'request': body, 'status': response.status_code, 'http_seconds': seconds}
                    data = response.json()
                    if response.status_code == 200:
                        record['response'] = {k: data[k] for k in ('model', 'answers', 'usage', 'id') if k in data}
                    else:
                        record['error_detail'] = data.get('detail', {})
                    save(dest, record)
                    if response.status_code not in (200, 400):
                        raise RuntimeError('Unexpected HTTP status ' + str(response.status_code))
                state_text = body['state'] if isinstance(body['state'], str) else json.dumps(body['state'], ensure_ascii=False, separators=(',', ':'))
                row['variants'][name] = {'status': record['status'],
                    'input_tokens': record.get('response', {}).get('usage', {}).get('input_tokens'),
                    'state_utf8_bytes': len(state_text.encode()), 'state_characters': len(state_text),
                    'http_seconds': record['http_seconds'], 'error_detail': record.get('error_detail')}
            rows.append(row)
            save(OUT / 'progress.json', {'completed_cases': len(rows), 'rows': rows})
            print(json.dumps({'case': n + 1, 'variants': {k: {'status': v['status'], 'tokens': v['input_tokens']} for k,v in row['variants'].items()}}), flush=True)
    summary = {}
    for name in ('original', 'compressed', 'readable'):
        values = [r['variants'][name] for r in rows]
        tokens = [v['input_tokens'] for v in values if v['input_tokens'] is not None]
        sizes = [v['state_utf8_bytes'] for v in values]
        summary[name] = {'http_status_counts': dict(collections.Counter(v['status'] for v in values)),
            'reported_tokens_count': len(tokens), 'reported_tokens_min': min(tokens) if tokens else None,
            'reported_tokens_median': statistics.median(tokens) if tokens else None,
            'reported_tokens_max': max(tokens) if tokens else None,
            'reported_tokens_sum': sum(tokens), 'state_bytes_min': min(sizes),
            'state_bytes_median': statistics.median(sizes), 'state_bytes_max': max(sizes)}
    save(OUT / 'comparison.json', {'status': 'completed', 'summary': summary, 'rows': rows})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
