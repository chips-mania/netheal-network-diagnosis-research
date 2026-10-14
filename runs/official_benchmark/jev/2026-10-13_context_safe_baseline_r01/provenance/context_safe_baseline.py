"""Readable-first Jev; lossless compression and partitioning for oversized evidence."""
import asyncio
import collections
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time
import readable_baseline_smoke as base
from jev_context_codec_v4 import encode_checked, decode, dumps
from jev_tree_partition_v4 import tree_partitions, restore

env = base.env
ROOT = base.ROOT
OUT = ROOT / 'runs/official_benchmark/jev/2026-10-13_context_safe_baseline_r01'
REPORT = ROOT / 'docs/research/official_benchmark/reports/jev_context_safe_baseline_20261013.md'
BUDGET = 24000  # Empirical routing guard, not the provider's token limit.


def prepare(state, purpose, options):
    clean = base.readable(state)
    clean['original_agent_system_prompt'] = base.SYSTEM
    clean['initial_user_message'] = (f"**Hint/Context**: {state['hint']}\n\n" if state.get('hint') else '') + (
        f"**Available Tools**: {', '.join(t['name'] for t in clean['tool_schemas'])}\n\n"
        'Begin by exploring the environment and gathering information to complete the task.')
    criteria = {f'c{i:03d}': v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for i,v in enumerate(options)}
    assert 1 <= len(criteria) <= 255
    q = {'decision': {'type': 'choice', 'instructions': base.ADAPTER + purpose, 'criteria': criteria}}
    assert clean['history'] == state['history']
    return clean, q


def pack(state, q, budget=BUDGET, force_compressed=False):
    available = budget - len(json.dumps(q, ensure_ascii=False).encode()) - 512
    assert available > 0, 'Question exceeds local envelope budget'
    raw = dumps(state)
    if not force_compressed and len(raw.encode()) <= available:
        return 'raw', [(raw, 1)]
    compact = encode_checked(state)
    if len(compact.encode()) <= available:
        return 'compressed', [(compact, 1)]
    chunks = tree_partitions(state, available)
    recovered = restore([f for c in chunks for f in c['evidence_fragments']])
    assert json.dumps(recovered, sort_keys=True) == json.dumps(state, sort_keys=True)
    return 'tree_partition', [(encode_checked(c), c['partition_weight']) for c in chunks]


class SafeJev(env.Jev):
    async def choose(self, state, purpose, options):
        clean, q = prepare(state, purpose, options)
        for budget in (BUDGET, 12000, 6000, 3000):
            mode, parts = pack(clean, q, budget, budget != BUDGET)
            answers = []
            for text, weight in parts:
                if time.monotonic() >= self.deadline:
                    raise TimeoutError('episode deadline')
                body = {'model': 'jev-1.13.0', 'state': text, 'questions': q}
                self.calls += 1
                env.append(self.folder/'request_started.jsonl', {'call': self.calls, 'unix': time.time()})
                start = time.perf_counter()
                response = await self.client.post('https://api.typesafe.ai/v1/systemone', json=body,
                    timeout=min(120, max(1, self.deadline-time.monotonic())))
                record = {'request': body, 'status': response.status_code, 'seconds': time.perf_counter()-start,
                          'mode': mode, 'weight': weight, 'guard_bytes': budget,
                          'original_state_sha256': hashlib.sha256(json.dumps(clean,sort_keys=True).encode()).hexdigest()}
                data = response.json()
                if response.status_code != 200:
                    record['error_detail'] = data.get('detail', {})
                    env.append(self.folder/'requests.jsonl', record)
                    if response.status_code == 400 and isinstance(record['error_detail'], dict) and record['error_detail'].get('error_type') == 'max_tokens_exceeded':
                        break  # Retry this decision with smaller, losslessly partitioned evidence.
                    raise RuntimeError('Jev HTTP ' + str(response.status_code))
                record['response'] = {k:data[k] for k in ('model','answers','usage','id') if k in data}
                env.append(self.folder/'requests.jsonl', record)
                answer = data['answers']['decision']
                assert answer['choice'] in q['decision']['criteria']
                assert set(answer['probabilities']) == set(q['decision']['criteria'])
                answers.append((answer,weight))
            else:
                if len(answers) == 1:
                    selected = list(q['decision']['criteria']).index(answers[0][0]['choice'])
                else:
                    scores = [sum(a['probabilities'][f'c{i:03d}']*w for a,w in answers)/sum(w for _,w in answers) for i in range(len(options))]
                    selected = max(range(len(options)), key=scores.__getitem__)
                env.append(self.folder/'decisions.jsonl', {'mode':mode,'parts':len(parts),'budget':budget,
                    'purpose':purpose,'selected':options[selected],'solver_turn':state['solver_turn']})
                return options[selected]
        raise env.ContextBudgetError('Exhausted lossless context fallback')


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def summarize(reference):
    rows = [read(p) for p in sorted((OUT/'episodes').glob('*/result.json'))]
    evaluator = env.CompetitionEvaluator()
    for r in rows:
        assert r['metrics']['scenario_fingerprint'] == reference[r['snapshot_id']]['scenario_fingerprint']
        evaluator.add_episode_metrics(env.EpisodeMetrics(**r['metrics']))
    report = {'status':'completed' if len(rows)==100 else 'running','evaluated':len(rows),
              'success':sum(r['metrics']['diagnosis_success'] for r in rows),
              'statuses':dict(collections.Counter(r['status'] for r in rows)),
              'official_summary':evaluator.compute_summary(),'episodes':rows}
    env.save(OUT/'summary.json',report)
    env.save(OUT/'manifest.json',{k:report[k] for k in ('status','evaluated','success','statuses')})
    return report


async def main():
    OUT.mkdir(parents=True,exist_ok=False)
    shutil.copy2(env.OUT/'tool_schemas.json',OUT/'tool_schemas.json')
    env.OUT=OUT
    env.Jev=SafeJev
    ref=read(env.UP/'manikyabard-20260201-100940.json')['results'][0]
    reference={r['snapshot_id']:r for a in ref['artifacts'] if a['label']=='aaa_metrics' for r in a['data']['results']}
    ss=env.snapshots()
    assert len(ss)==len(reference)==100
    for s in ss:
        e,w,_=env.runtime(s)
        assert w._trace.scenario_fingerprint==reference[s['snapshot_id']]['scenario_fingerprint']
        e.close()
    # Offline regression on every actual failed request; no evidence truncation.
    prior=ROOT/'runs/official_benchmark/jev/2026-10-13_readable_baseline_full_r01/episodes'
    checked=[]
    for p in sorted(prior.glob('*/requests.jsonl')):
        for line in p.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['status']!=400:
                continue
            state=json.loads(r['request']['state']); q=r['request']['questions']
            mode,parts=pack(state,q)
            if mode=='compressed':
                assert json.dumps(decode(json.loads(parts[0][0])),sort_keys=True)==json.dumps(state,sort_keys=True)
            checked.append({'case':p.parent.name,'mode':mode,'parts':len(parts)})
    assert len(checked)==40
    env.save(OUT/'preflight.json',{'fingerprints':100,'failed_inputs_checked':40,'results':checked})
    sources=[Path(__file__),Path(base.__file__),Path(env.__file__),ROOT/'experiments/official_benchmark/jev/compare_first_case_input.py',
             ROOT/'netheal-topology-generalization/scripts/jev_context_codec_v4.py',ROOT/'netheal-topology-generalization/scripts/jev_tree_partition_v4.py']
    env.save(OUT/'protocol.json',{'model':'jev-1.13.0','cases':100,'fresh_runs':100,'concurrency':1,
        'routing':'readable <= local 24KB envelope, else lossless compression, else JSON-tree partitions',
        'fallback_budgets':[24000,12000,6000,3000],'fallback_trigger':'API max_tokens_exceeded only',
        'aggregation':'value-byte-weighted probabilities; single part provider choice',
        'max_steps':100,'max_turns':100,'deadline_seconds':1200,'fixed_wait':0,
        'source_hashes':{str(p.relative_to(ROOT)):env.sha(p) for p in sources},
        'limitation':'Compression preserves reconstructability, not model comprehension. Partition aggregation is not joint-context inference.'})
    provenance=OUT/'provenance'; provenance.mkdir()
    for p in sources: shutil.copy2(p,provenance/p.name)
    key=env.dotenv_values(ROOT/'.env').get('TYPESAFE_API_KEY'); assert key
    # Probe two historical failing bodies using the frozen routing; do not use predictions for tuning.
    for item in checked[:2]:
        folder=OUT/'acceptance_checks'/item['case']; folder.mkdir(parents=True)
        prior_records=[json.loads(x) for x in (prior/item['case']/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        old=next(r['request'] for r in prior_records if r['status']==400)
        state=json.loads(old['state'])
        purpose=old['questions']['decision']['instructions'].removeprefix(base.ADAPTER)
        options=list(old['questions']['decision']['criteria'].values())
        client=SafeJev(key,folder,time.monotonic()+1200)
        try: await client.choose(state,purpose,options)
        finally: await client.client.aclose()
    print('PREFLIGHT: 40 failed inputs preserved; 2 API acceptance probes passed',flush=True)
    for s in ss:
        with (OUT/'execution.log').open('a',encoding='utf-8') as log,contextlib.redirect_stdout(log):
            await env.episode(s,key,OUT/'episodes'/s['snapshot_id'])
        report=summarize(reference)
        print(json.dumps({k:report[k] for k in ('evaluated','success','statuses')}),flush=True)
    req=[json.loads(x) for p in (OUT/'episodes').glob('*/requests.jsonl') for x in p.read_text(encoding='utf-8').splitlines()]
    dec=[json.loads(x) for p in (OUT/'episodes').glob('*/decisions.jsonl') for x in p.read_text(encoding='utf-8').splitlines()]
    times=[r['seconds'] for r in report['episodes']]
    tokens=sum(r.get('response',{}).get('usage',{}).get('input_tokens',0) for r in req)
    report['usage']={'requests':len(req),'input_tokens':tokens,'estimated_usd':tokens*.042/1e6,
        'api_seconds':sum(r['seconds'] for r in req),'http_statuses':dict(collections.Counter(r['status'] for r in req)),
        'decision_modes':dict(collections.Counter(r['mode'] for r in dec)),
        'episode_min_seconds':min(times),'episode_mean_seconds':statistics.mean(times),'episode_max_seconds':max(times)}
    env.save(OUT/'summary.json',report)
    REPORT.write_text('# Jev 원문 우선·문맥 한도 처리 재평가 — 2026-10-13\n\n'
        '동일 공식 100개를 모두 새로 실행했다. Claude는 공개 결과만 재사용했다. 원문 입력과 원본 지시를 유지하되, 로컬 24KB 가드를 넘으면 기존 무손실 압축, 그래도 길면 JSON 경로 분할 및 원자료 값 바이트 가중 확률 평균을 적용했다. 24KB는 공식 토큰 한도가 아닌 보수적인 라우팅 기준이다. 관측 삭제·숫자 반올림·정답 기반 선별은 하지 않았다.\n\n'
        f"정답 {report['success']}/100. 상태: `{report['statuses']}`. 공식 가중 성공률 {report['official_summary']['diagnosis_success_rate']:.2%}. Claude 공개 기준선은 66/100, 가중 65.04%다.\n\n"
        f"시간·비용·표현별 판단 횟수:\n\n```json\n{json.dumps(report['usage'],ensure_ascii=False,indent=2)}\n```\n\n"
        '비용은 최종 평가 요청의 보고 usage만 포함하며 사전 수용 검사 비용은 제외한다. 원본 Claude와 선택형 인터페이스·실행 환경이 다르며, 압축의 복원 가능성이 추론 성능 보존을 의미하지 않는다. 분할은 전체 증거를 한 번에 읽는 것과 다르다. 이미 관측한 평가셋의 후속 실험이고, 단일 실행이다. 정확도 차이를 모델만의 차이로 주장하지 않는다.\n\n'
        '자료: `runs/official_benchmark/jev/2026-10-13_context_safe_baseline_r01/`의 `summary.json`, `protocol.json`, `preflight.json`, `provenance/`, `episodes/`, `acceptance_checks/`.\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'success':report['success'],'usage':report['usage']},indent=2),flush=True)


if __name__=='__main__': asyncio.run(main())
