"""Single-state Jev evaluation against the published Claude reference."""
import asyncio
import collections
import contextlib
import json
import os
from pathlib import Path
import shutil
import statistics
import time

import readable_baseline_smoke as baseline
from compare_first_case_input import simplify_schema
from structured_state import StructuredState, dumps, measurement

env = baseline.env
ROOT = baseline.ROOT
OUT = ROOT / 'runs/official_benchmark/jev/2026-10-13_structured_state_r01'
REPORT = ROOT / 'docs/research/official_benchmark/reports/jev_structured_state_20261013.md'


class StructuredJev(env.Jev):
    def __init__(self, *args):
        super().__init__(*args)
        self.reducer = StructuredState()

    async def choose(self, state, purpose, options):
        clean = self.reducer.render(state)
        clean['tool_schemas'] = [{'name': t['name'], 'description': t['description'],
                                 'inputSchema': simplify_schema(t['inputSchema'])}
                                for t in state['tool_schemas']]
        clean['original_agent_system_prompt'] = baseline.SYSTEM
        if self.reducer.latest is not None:
            last = next(e for e in reversed(state['history']) if 'result' in e)
            assert clean['latest_observation']['observation'] == measurement(last)
            assert clean['latest_observation']['arguments'] == last['arguments']
        criteria = {f'c{i:03d}': v if isinstance(v, str) else dumps(v) for i, v in enumerate(options)}
        assert 1 <= len(criteria) <= 255
        body = {'model': 'jev-1.13.0', 'state': dumps(clean), 'questions': {'decision': {
            'type': 'choice', 'instructions': baseline.ADAPTER + purpose, 'criteria': criteria}}}
        if time.monotonic() >= self.deadline:
            raise TimeoutError('episode deadline')
        self.calls += 1
        # Persist the exact request before sending, including failed network attempts.
        env.append(self.folder / 'request_started.jsonl', {'call': self.calls, 'unix': time.time(), 'request': body})
        started = time.perf_counter()
        response = await self.client.post('https://api.typesafe.ai/v1/systemone', json=body,
            timeout=min(120, max(1, self.deadline-time.monotonic())))
        record = {'request': body, 'status': response.status_code,
                  'seconds': time.perf_counter()-started, 'latest_measurement_verified': True,
                  'raw_history_state_bytes': len(dumps(state).encode()),
                  'structured_state_bytes': len(body['state'].encode())}
        try:
            data = response.json()
        except ValueError:
            data = {'invalid_json_response': True}
        record['response'] = data
        env.append(self.folder / 'requests.jsonl', record)
        if response.status_code != 200:
            raise RuntimeError('Jev HTTP ' + str(response.status_code))
        choice = data['answers']['decision']['choice']
        assert choice in criteria
        return options[list(criteria).index(choice)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def summarize(reference):
    rows = [read(p) for p in sorted((OUT / 'episodes').glob('*/result.json'))]
    evaluator = env.CompetitionEvaluator()
    pairs = collections.Counter()
    for row in rows:
        m = row['metrics']
        ref = reference[row['snapshot_id']]
        assert m['scenario_fingerprint'] == ref['scenario_fingerprint']
        evaluator.add_episode_metrics(env.EpisodeMetrics(**m))
        a, b = m['diagnosis_success'], ref['pass']
        pairs['both' if a and b else 'jev_only' if a else 'claude_only' if b else 'neither'] += 1
    result = {'status': 'completed' if len(rows) == len(reference) else 'running',
              'evaluated': len(rows), 'total': len(reference),
              'success': sum(r['metrics']['diagnosis_success'] for r in rows),
              'statuses': dict(collections.Counter(r['status'] for r in rows)),
              'paired_published_claude': dict(pairs),
              'official_summary': evaluator.compute_summary(), 'episodes': rows}
    env.save(OUT / 'summary.json', result)
    return result


async def main():
    key = os.environ.get('TYPESAFE_API_KEY') or env.dotenv_values(ROOT / '.env').get('TYPESAFE_API_KEY')
    assert key, 'TYPESAFE_API_KEY missing'
    ref = read(env.UP / 'manikyabard-20260201-100940.json')['results'][0]
    reference = {r['snapshot_id']: r for a in ref['artifacts'] if a['label'] == 'aaa_metrics' for r in a['data']['results']}
    snapshots = env.snapshots()
    assert len(snapshots) == len(reference) == 100
    for s in snapshots:
        e, w, _ = env.runtime(s)
        assert w._trace.scenario_fingerprint == reference[s['snapshot_id']]['scenario_fingerprint']
        e.close()
    sources = [Path(__file__), Path(__file__).with_name('structured_state.py'), Path(env.__file__),
               Path(baseline.__file__), baseline.CLAUDE_SOURCE,
               Path(__file__).with_name('compare_first_case_input.py')]
    hashes = {str(p.relative_to(ROOT)): env.sha(p) for p in sources}
    if OUT.exists():
        assert read(OUT / 'protocol.json')['source_hashes'] == hashes
    else:
        OUT.mkdir(parents=True)
        shutil.copy2(env.OUT / 'tool_schemas.json', OUT / 'tool_schemas.json')
        env.save(OUT / 'protocol.json', {
            'source_hashes': hashes, 'model': 'jev-1.13.0', 'cases': 100,
            'snapshot_ids': [s['snapshot_id'] for s in snapshots], 'fingerprints_verified': 100,
            'compression': 'decision-oriented structured state; not strict lossless',
            'partition': False, 'weighted_score_aggregation': False, 'retries': 0,
            'concurrency': 1, 'max_turns': 100, 'max_environment_steps': 100,
            'deadline_seconds': 1200, 'repeat_actions_allowed': True,
            'candidate_selection': 'Same tool then argument choices as readable baseline; each choice sees one complete state.',
            'raw_results': 'Every tool result and helper result retained verbatim in JSONL.',
            'omissions': ['repeated observation snapshots', 'hashed RL device/diagnostic arrays',
                          'reward and reward_breakdown', 'full valid_actions arrays (provided as relevant candidates)'],
            'evidence': 'All action_result fields retained; identical consecutive measurements coalesced with every event/step/timestamp.',
            'reference': 'Published Claude 100 results; no Claude rerun; input formats differ.',
            'reference_sha256': env.sha(env.UP / 'manikyabard-20260201-100940.json'),
            'interruption_policy': 'Completed cases reused; incomplete directories require explicit recovery; no outcome-based reruns.',
            'frozen_unix': time.time()})
        for p in sources:
            dest = OUT / 'provenance' / p.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(p, dest)
    env.OUT, env.Jev = OUT, StructuredJev
    for s in snapshots:
        folder = OUT / 'episodes' / s['snapshot_id']
        if (folder / 'result.json').exists():
            continue
        assert not folder.exists(), 'Incomplete episode requires explicit recovery'
        with (OUT / 'execution.log').open('a', encoding='utf-8') as log, contextlib.redirect_stdout(log):
            await env.episode(s, key, folder)
        result = summarize(reference)
        print(dumps({k: result[k] for k in ('evaluated', 'success', 'statuses')}), flush=True)
    result = summarize(reference)
    requests = [json.loads(line) for p in (OUT / 'episodes').glob('*/requests.jsonl')
                for line in p.read_text(encoding='utf-8').splitlines()]
    tokens = [r['response'].get('usage', {}).get('input_tokens', 0) for r in requests]
    result['usage'] = {'requests': len(requests), 'reported_input_tokens': sum(tokens),
        'max_reported_input_tokens': max(tokens, default=0),
        'http_statuses': dict(collections.Counter(r['status'] for r in requests)),
        'api_seconds': sum(r['seconds'] for r in requests),
        'mean_episode_seconds': statistics.mean(r['seconds'] for r in result['episodes']),
        'structured_state_bytes': sum(r['structured_state_bytes'] for r in requests),
        'same_trajectory_raw_history_bytes': sum(r['raw_history_state_bytes'] for r in requests)}
    env.save(OUT / 'summary.json', result)
    official = result['official_summary']
    usage = result['usage']
    by_type = []
    for ft in sorted({r['metrics']['ground_truth_type'] for r in result['episodes']}):
        group = [r for r in result['episodes'] if r['metrics']['ground_truth_type'] == ft]
        by_type.append(f"| {ft} | {len(group)} | {sum(r['metrics']['diagnosis_success'] for r in group)} | {sum(reference[r['snapshot_id']]['pass'] for r in group)} |")
    REPORT.write_text(f'''# JEV 구조화 상태 실험 — 2026-10-13

공개 Claude와 같은 100개 초기 사례를 사용했고 fingerprint 100/100을 확인했다. Claude는 재실행하지 않았다.

| 지표 | JEV 구조화 상태 | 공개 Claude |
|---|---:|---:|
| 유형·위치 공동 성공 | {result['success']}/100 | 66/100 |
| 공식 장비 수 가중 성공률 | {official['diagnosis_success_rate']:.2%} | 65.04% |
| 유형 macro-F1 | {official['fault_type_macro_f1']:.4f} | 0.6373 |
| 공식 가중 평균 단계 | {official['avg_steps']:.2f} | 19.01 |

공식 평가 지표의 장비 수 가중치는 모델 판단 점수의 가중평균과 별개다. 행동 선택에는 분할 입력이나 점수 가중평균을 사용하지 않았다.

| 실제 유형 | 사례 수 | JEV 성공 | Claude 성공 |
|---|---:|---:|---:|
{chr(10).join(by_type)}

실행 상태: `{result['statuses']}`. 사례별 비교: `{result['paired_published_claude']}`. 모든 오류와 단계 소진 사례도 100개 분모에 포함했다.

원본 도구 결과·후보 조회·API 요청/응답을 보존했다. 결정론적 reducer는 공개 action_result의 수치, 대상, 성공 여부, 시각을 유지하며 동일 측정은 횟수와 관측 시점으로 묶는다. 변경된 측정은 전부 보존한다. 장치 상태 0을 정상으로 해석하지 않는다. 행렬은 원래 인덱스와 연결 값을 유지하며 비공개 장치 매핑을 읽지 않는다. 보상, 반복 스냅샷, 해시 기반 RL 배열을 생략하므로 strict lossless가 아니다. 장애 원인 추론·후보 우선순위·반복 금지는 없다.

한 선택 질문당 전체 구조화 상태를 한 번 전달한다. 도구와 인자는 기존 방식대로 순차 선택하며 결과 점수를 합치지 않는다. 모델 jev-1.13.0, 100 tool steps/100 solver turns, 사례당 1200초, 동시성 1, 재시도 0이다.

API 요청 {usage['requests']:,}회, 보고 입력 {usage['reported_input_tokens']:,}토큰, 최대 {usage['max_reported_input_tokens']:,}토큰. 평균 사례 시간 {usage['mean_episode_seconds']:.2f}초. 동일 실행 경로의 원본 history 상태 합계 {usage['same_trajectory_raw_history_bytes']:,}바이트 대비 실제 구조화 상태 {usage['structured_state_bytes']:,}바이트다. 이 바이트 비교는 토큰 감소율과 동일하지 않다. Claude 토큰 비용과 직접적인 속도 배수는 산정하지 않는다.

기존 원문 JEV 결과는 36/100, max_tokens_exceeded 40건이었다. 이번 비교는 이미 관측한 평가셋의 후속 실험이며 입력 표현·정보 생략 효과가 함께 포함된다. 공개 Claude는 원래 대화 이력/native tool calling을 사용했으므로 동일 입력을 통제한 모델 우열 검증은 아니다.

원시 자료: `runs/official_benchmark/jev/2026-10-13_structured_state_r01/`. 프로토콜과 소스 해시는 실행 전에 고정했다.
''', encoding='utf-8')
    print(dumps({'status': result['status'], 'success': result['success'], 'usage': usage}), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
