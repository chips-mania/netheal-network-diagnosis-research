"""Frozen readable Jev evaluation on all 100 published Claude snapshots."""
import asyncio
import collections
import contextlib
import json
import shutil
import statistics
import time
from pathlib import Path
import readable_baseline_smoke as smoke

env = smoke.env
ROOT = smoke.ROOT
OUT = ROOT / 'runs/official_benchmark/jev/2026-10-13_readable_baseline_full_r01'
REPORT = ROOT / 'docs/research/official_benchmark/reports/jev_readable_baseline_full_20261013.md'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def summarize(reference):
    rows = [read(p) for p in sorted((OUT / 'episodes').glob('*/result.json'))]
    evaluator = env.CompetitionEvaluator()
    paired = collections.Counter()
    for row in rows:
        m = row['metrics']
        assert m['scenario_fingerprint'] == reference[row['snapshot_id']]['scenario_fingerprint']
        evaluator.add_episode_metrics(env.EpisodeMetrics(**m))
        a, b = m['diagnosis_success'], reference[row['snapshot_id']]['pass']
        paired['both' if a and b else 'jev_only' if a else 'claude_only' if b else 'neither'] += 1
    status = collections.Counter(r['status'] for r in rows)
    report = {'status': 'completed' if len(rows) == 100 else 'running', 'evaluated': len(rows),
              'total': 100, 'success': sum(r['metrics']['diagnosis_success'] for r in rows),
              'statuses': dict(status), 'paired_claude': dict(paired),
              'official_summary': evaluator.compute_summary(), 'episodes': rows}
    env.save(OUT / 'summary.json', report)
    env.save(OUT / 'manifest.json', {k: report[k] for k in ('status', 'evaluated', 'total', 'success', 'statuses')})
    return report


async def main():
    original_out = env.OUT
    source_paths = [Path(__file__), Path(smoke.__file__), Path(env.__file__),
                    ROOT / 'experiments/official_benchmark/jev/compare_first_case_input.py', smoke.CLAUDE_SOURCE]
    hashes = {str(p.relative_to(ROOT)): env.sha(p) for p in source_paths}
    ref = read(env.UP / 'manikyabard-20260201-100940.json')['results'][0]
    reference = {r['snapshot_id']: r for a in ref['artifacts'] if a['label'] == 'aaa_metrics' for r in a['data']['results']}
    snapshots = env.snapshots()
    assert len(snapshots) == len(reference) == 100
    for snapshot in snapshots:
        e, w, _ = env.runtime(snapshot)
        assert w._trace.scenario_fingerprint == reference[snapshot['snapshot_id']]['scenario_fingerprint']
        e.close()
    if OUT.exists():
        assert read(OUT / 'protocol.json')['source_hashes'] == hashes
    else:
        OUT.mkdir(parents=True)
        shutil.copy2(original_out / 'tool_schemas.json', OUT / 'tool_schemas.json')
        protocol = read(smoke.OUT / 'protocol.json')
        protocol.pop('case')
        protocol.update({'cases': 100, 'source_hashes': hashes, 'fingerprints_verified': 100,
                         'first_case_reused': str(smoke.OUT.relative_to(ROOT)),
                         'selection_rule': 'Reuse the only completed smoke; execute every other snapshot once regardless of outcome.',
                         'interruption_policy': 'Preserve incomplete directories; do not silently restart.',
                         'frozen_unix': time.time()})
        env.save(OUT / 'protocol.json', protocol)
        provenance = OUT / 'provenance'
        provenance.mkdir()
        for path in source_paths:
            shutil.copy2(path, provenance / path.name)
        destination = OUT / 'episodes' / smoke.SID
        shutil.copytree(smoke.OUT / 'episode', destination)
        env.save(destination / 'reuse.json', {'source': str(smoke.OUT / 'episode'), 'reason': 'same configuration; no rerun after observing success'})
    env.OUT = OUT
    env.Jev = smoke.ReadableJev
    key = env.dotenv_values(ROOT / '.env').get('TYPESAFE_API_KEY')
    assert key
    start = time.perf_counter()
    for snapshot in snapshots:
        folder = OUT / 'episodes' / snapshot['snapshot_id']
        if (folder / 'result.json').exists():
            continue
        assert not folder.exists(), 'Interrupted case preserved; explicit recovery required'
        # Preserve detailed console records without flooding the caller.
        with (OUT / 'execution.log').open('a', encoding='utf-8') as log, contextlib.redirect_stdout(log):
            await env.episode(snapshot, key, folder)
        report = summarize(reference)
        print(json.dumps({k: report[k] for k in ('evaluated', 'success', 'statuses')}), flush=True)
    report = summarize(reference)
    requests = [json.loads(line) for path in (OUT / 'episodes').glob('*/requests.jsonl')
                for line in path.read_text(encoding='utf-8').splitlines()]
    tokens = [r.get('response', {}).get('usage', {}).get('input_tokens', 0) for r in requests]
    usage = {'recorded_requests': len(requests), 'input_tokens': sum(tokens), 'estimated_usd': sum(tokens)*.042/1e6,
             'max_reported_input_tokens': max(tokens), 'http_statuses': dict(collections.Counter(r['status'] for r in requests)),
             'error_types': dict(collections.Counter(r.get('error_detail', {}).get('error_type', 'unspecified')
                            for r in requests if r['status'] != 200)),
             'api_seconds': sum(r['seconds'] for r in requests),
             'mean_episode_seconds': statistics.mean(r['seconds'] for r in report['episodes']),
             'median_episode_seconds': statistics.median(r['seconds'] for r in report['episodes']),
             'this_invocation_seconds_excluding_preflight': time.perf_counter()-start,
             'includes_reused_first_case_usage': True}
    report['usage'] = usage
    env.save(OUT / 'summary.json', report)
    types = []
    for ft in sorted({r['metrics']['ground_truth_type'] for r in report['episodes']}):
        group = [r for r in report['episodes'] if r['metrics']['ground_truth_type'] == ft]
        types.append(f"| {ft} | {len(group)} | {sum(r['metrics']['diagnosis_success'] for r in group)} | {sum(reference[r['snapshot_id']]['pass'] for r in group)} |")
    official = report['official_summary']
    text = f'''# Jev 원문 구조 전체 100개 평가 — 2026-10-13

연구자 공개 Claude 결과를 기준선으로 사용하고 Jev만 실행했다. 동일 초기 스냅샷 fingerprint 100/100 일치. 첫 smoke는 같은 설정의 결과를 재사용했고 나머지 99개는 한 번씩 실행했다. 오류·오진을 이유로 재실행하거나 제외하지 않았다.

## 결과

| 지표 | Jev 원문 구조 | 공개 Claude |
|---|---:|---:|
| 공동 진단 성공 | {report['success']}/100 | 66/100 |
| 공식 장비 수 가중 성공률 | {official['diagnosis_success_rate']:.2%} | 65.04% |
| 유형 macro-F1 | {official['fault_type_macro_f1']:.4f} | 0.6373 |
| 공식 가중 평균 단계 | {official['avg_steps']:.2f} | 19.01 |
| 단순 평균 에피소드 시간 | {usage['mean_episode_seconds']:.2f}초 | 221.72초 |

실행 상태: `{report['statuses']}`. API 상태: `{usage['http_statuses']}`. 오류 유형: `{usage['error_types']}`.

| 실제 유형 | 사례 수 | Jev 성공 | Claude 성공 |
|---|---:|---:|---:|
{chr(10).join(types)}

사례별 공동 결과: `{report['paired_claude']}`.

## 입력 및 해석

첫 사례 smoke와 동일한 일반 JSON 문자열, 전체 공개 관측 이력, 간소화 도구 정의, 원본 Claude SYSTEM_PROMPT와 선택형 응답 변환 지시를 사용했다. 압축·분할·24KB 가드·고정 대기 없음. 모델은 jev-1.13.0, 동시성 1, 환경 100단계/solver 100 turn/1200초, 재시도 없음. 모델 정답과 비공개 상태는 입력하지 않았다.

원본 Claude의 native tool calling과 생성 설명문은 Jev 선택형 도구·인자 질문으로 변환됐다. 과거 실제 대화를 재생한 실험이 아니며, 이후 관측은 각 에이전트 행동에 따라 다르다. Windows 직접 handler와 원본 컨테이너의 실행 환경 차이도 남는다. 시간 차이는 모델 자체의 속도 배수로 해석하지 않는다. 실패 사례의 조기 종료도 전체 평균 시간에 영향을 준다.

입력 수용 실패와 진단 오답을 구분하되 모든 100개를 성능 분모에 포함한다. 기존 평가에서 이미 관측한 사례를 재사용한 후속 평가이며 새로운 미관측 평가셋이 아니다. 압축 버전과의 차이는 입력 표현뿐 아니라 지시 및 실행 설정 변경을 포함하므로 압축 제거 하나의 인과 효과로 주장하지 않는다.

## 시간·비용과 기록

API 기록 {usage['recorded_requests']:,}회, 보고 입력 {usage['input_tokens']:,}토큰, HTTP 응답 시간 합계 {usage['api_seconds']:.2f}초. 보고 usage와 입력 100만 토큰당 $0.042 기준 예상 비용 ${usage['estimated_usd']:.6f}. usage 없는 실패 요청 비용은 산정할 수 없으며 청구 확정액이 아니다. Claude 실제 토큰·비용은 확보하지 못해 비용 절감 배수는 계산하지 않는다. Python 시작·import·preflight는 에피소드 시간 밖이다.

원시 자료: `runs/official_benchmark/jev/2026-10-13_readable_baseline_full_r01/`의 `summary.json`, `protocol.json`, `provenance/`, `episodes/`. 각 사례의 요청·응답·도구·후보 조회·종료 지표를 보존했다. 실행 스크립트: `experiments/official_benchmark/jev/readable_baseline_full.py`.
'''
    REPORT.write_text(text, encoding='utf-8')
    print(json.dumps({'status': report['status'], 'success': report['success'], 'usage': usage}, indent=2), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
