"""Validate and compare complete official results without excluding failed cases."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'runs/official_benchmark/laya/2026-10-12_official_baseline_r01'
JEV = ROOT / 'runs/official_benchmark/jev/netheal_official_jev_20261011_v4'
UP = ROOT / 'netheal-topology-generalization/results/netheal_official_jev_20261011_v1/upstream'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    import sys
    sys.path.insert(0, str(ROOT / 'netheal-topology-generalization/scripts'))
    from jev_context_codec_v4 import decode
    from jev_tree_partition_v4 import restore
    from jev_official_eval_v4 import CompetitionEvaluator, EpisodeMetrics
    rows = read(OUT / 'summary.json')['episodes']
    assert len(rows) == len({r['snapshot_id'] for r in rows}) == 100
    jev = {r['snapshot_id']: r for r in read(JEV / 'summary.json')['episodes']}
    ref = read(UP / 'manikyabard-20260201-100940.json')['results'][0]
    claude = {r['snapshot_id']: r for a in ref['artifacts'] if a['label'] == 'aaa_metrics' for r in a['data']['results']}
    assert {r['snapshot_id'] for r in rows} == set(jev) == set(claude)
    evaluator = CompetitionEvaluator()
    total_requests = total_questions = total_tokens = 0
    modes = collections.Counter()
    verification = []
    for r in rows:
        sid = r['snapshot_id']
        assert r['metrics']['scenario_fingerprint'] == jev[sid]['metrics']['scenario_fingerprint'] == claude[sid]['scenario_fingerprint']
        assert r['metrics']['steps'] <= 100
        evaluator.add_episode_metrics(EpisodeMetrics(**r['metrics']))
        attempts = [p for p in (OUT / 'episodes' / sid).glob('attempt_*/result.json') if read(p) == r]
        assert len(attempts) == 1
        folder = attempts[0].parent
        requests = [json.loads(x) for x in (folder / 'requests.jsonl').read_text(encoding='utf-8').splitlines()] if (folder / 'requests.jsonl').exists() else []
        decisions = [json.loads(x) for x in (folder / 'decisions.jsonl').read_text(encoding='utf-8').splitlines()] if (folder / 'decisions.jsonl').exists() else []
        groups = collections.defaultdict(list)
        for request in requests:
            state = decode(json.loads(request['request']['state']))
            assert hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest() == request['raw_state_sha256']
            assert request['tokens'] <= 8192
            assert len(request['request']['state'].encode()) + len(json.dumps(request['request']['questions'], ensure_ascii=False).encode()) + 512 <= 24000
            probs = request['response']['answers']['decision']['probabilities']
            assert set(probs) == set(request['request']['questions']['decision']['criteria'])
            assert abs(sum(probs.values()) - 1.) < 1e-5
            if 'evidence_fragments' in state:
                groups[request['question']].append(state)
        completed_questions = {d['question'] for d in decisions}
        for qid, group in groups.items():
            if qid not in completed_questions:
                continue  # timed out during a multi-fragment decision; retained as incomplete.
            original = restore([f for g in group for f in g['evidence_fragments']])
            assert hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest() == group[0]['original_state_sha256']
        total_requests += len(requests)
        total_questions += len(decisions)
        total_tokens += sum(x['tokens'] for x in requests)
        modes.update(x['mode'] for x in decisions)
        verification.append({'snapshot_id': sid, 'requests': len(requests), 'questions': len(decisions), 'status': r['status']})
    official = evaluator.compute_summary()
    assert official == read(OUT / 'summary.json')['official_summary']
    success = sum(r['metrics']['diagnosis_success'] for r in rows)
    statuses = collections.Counter(r['status'] for r in rows)
    weighted = sum(r['metrics']['network_size'] * r['metrics']['diagnosis_success'] for r in rows) / sum(r['metrics']['network_size'] for r in rows)
    assert abs(weighted - official['diagnosis_success_rate']) < 1e-12
    paired = {name: collections.Counter('both' if a and b else 'laya_only' if a else name + '_only' if b else 'neither'
        for r in rows for a, b in [(r['metrics']['diagnosis_success'], jev[r['snapshot_id']]['metrics']['diagnosis_success'] if name == 'jev' else claude[r['snapshot_id']]['pass'])]) for name in ('jev', 'claude')}
    with (OUT / 'paired_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['snapshot_id', 'ground_truth_type', 'laya_success', 'jev_success', 'claude_success', 'laya_status', 'laya_steps'])
        for r in rows:
            sid = r['snapshot_id']
            writer.writerow([sid, r['metrics']['ground_truth_type'], r['metrics']['diagnosis_success'], jev[sid]['metrics']['diagnosis_success'], claude[sid]['pass'], r['status'], r['metrics']['steps']])
    (OUT / 'verification.json').write_text(json.dumps({'status': 'passed', 'episodes': verification, 'requests': total_requests,
        'questions': total_questions, 'tokens': total_tokens, 'partition_modes': modes, 'paired': paired}, indent=2), encoding='utf-8')
    j = read(JEV / 'summary.json')['official_summary']
    cw = sum(c['network_size'] * c['pass'] for c in claude.values()) / sum(c['network_size'] for c in claude.values())
    text = f'''# LAYA 기본 모델 공식 100개 평가

추가 학습 0회, 원본 100개 snapshot/fingerprint 일치. 전체 100개를 분모에 포함했다.

| 지표 | Claude 원본 | Jev v4 | 기본 LAYA |
|---|---:|---:|---:|
| 유형·위치 공동 성공 | {sum(c['pass'] for c in claude.values())}/100 | {sum(r['metrics']['diagnosis_success'] for r in jev.values())}/100 | {success}/100 |
| 공식 장비 수 가중 성공률 | {cw:.2%} | {j['diagnosis_success_rate']:.2%} | {weighted:.2%} |
| 유형 macro-F1 | 0.6373 | {j['fault_type_macro_f1']:.4f} | {official['fault_type_macro_f1']:.4f} |

LAYA 단순 유형 정확도: {sum(r['metrics']['predicted_type'] == r['metrics']['ground_truth_type'] for r in rows)/100:.2%}.
종료 상태: `{dict(statuses)}`. 완료 진단 중 오진: {sum(r['status'] == 'completed' and not r['metrics']['diagnosis_success'] for r in rows)}개.
완료 판단 질문 {total_questions}개, 저장된 조각 forward {total_requests}개, 처리 토큰 {total_tokens:,}개.
분할 규칙: `{dict(modes)}`. 오류/시간초과 도중 시작했으나 응답이 없는 forward는 request_started 로그에 별도 보존한다.

## 비교 해석

원본 Claude는 기준 결과를 재사용했다. LAYA는 동결 Jev v4의 전체 episode 함수를 사용하고 모델 client만 대체했다.
도구·인자 후보 및 분기, 환경·채점·100단계/100 turn/1200초 제한은 유지했다.
모델 행동에 따라 후속 관측은 달라진다. LAYA 토큰 한도에 따른 추가 분할, 기본 checkpoint 온도 보정,
FP32 단일 로컬 GPU와 Windows 실행 환경의 차이가 있어 순수 모델 교체 효과라고 단정하지 않는다.
이 100개는 이미 프로젝트에서 분석한 공식 사례이며 새 미관측 평가셋이 아니다.

기본 LAYA의 도구 정책 전체 결과이며 고정 관측 진단 정확도나 학습 후 능력 상한을 뜻하지 않는다.
정답률에 따라 사례를 제외하거나 재실행하지 않았다. 학습·Jev/Claude API 재호출은 하지 않았다.

사례별 비교: `{dict(paired)}`.

## 산출물

- 실행 기록: `runs/official_benchmark/laya/2026-10-12_official_baseline_r01/`
- 집계: `summary.json`, `verification.json`, `paired_results.csv`
- 설정/출처: `frozen_protocol.json`, `provenance/`, `preflight.json`
- 원시 기록: `episodes/*/attempt_*/requests.jsonl`, `decisions.jsonl`, `tools.jsonl`, `result.json`
'''
    report = ROOT / 'docs/research/official_benchmark/reports/laya_official_baseline_20261012.md'
    report.write_text(text, encoding='utf-8')
    print(json.dumps({'success': success, 'weighted': weighted, 'statuses': statuses, 'report': str(report)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
