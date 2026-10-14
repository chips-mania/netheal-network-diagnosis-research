"""Verify frozen v3 artifacts and report all 100 cases without filtering errors."""
import collections, hashlib, json, statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from jev_context_codec import decode

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'netheal-topology-generalization/results'
OUT=BASE/'netheal_official_jev_20261011_v3'
V2=BASE/'netheal_official_jev_20261011_v2'
V1=BASE/'netheal_official_jev_20261011_v1'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    return (json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def audit(x):
    if isinstance(x,dict):
        assert not set(x)&{'ground_truth_fault','ground_truth','fault_info','injected_fault','correct_diagnosis','expected_solution'}
        for v in x.values(): audit(v)
    elif isinstance(x,list):
        for v in x: audit(v)
def outcome(r):
    if r['metrics']['diagnosis_success']: return 'success'
    if r['status']=='completed': return 'wrong_diagnosis'
    return r['status']

summary=read(OUT/'summary.json'); episodes=summary['episodes']; official=summary['official_summary']
old_summary=read(V2/'summary.json'); old={r['snapshot_id']:r for r in old_summary['episodes']}
ref=read(V1/'upstream/manikyabard-20260201-100940.json')['results'][0]
claude={r['snapshot_id']:r for a in ref['artifacts'] if a['label']=='aaa_metrics' for r in a['data']['results']}
assert len(episodes)==100 and len({r['snapshot_id'] for r in episodes})==100
assert set(old)==set(claude)=={r['snapshot_id'] for r in episodes}
for r in episodes: assert r['metrics']['scenario_fingerprint']==claude[r['snapshot_id']]['scenario_fingerprint']
protocol=read(OUT/'frozen_protocol.json')
for name,key in [('jev_official_eval_v3.py','script_sha256'),('jev_context_codec.py','codec_sha256')]:
    assert sha(ROOT/'netheal-topology-generalization/scripts'/name)==protocol[key]==sha(OUT/name)
assert sha(OUT/'execution_spec.md')==protocol['spec_sha256']

costs={}; full=collections.Counter(); api_seconds=0; max_tokens=0; max_guard=0; model_ids=set(); exact=0
first_unix=[]
for f in OUT.rglob('request_started.jsonl'):
    first_unix.extend(r['unix'] for r in rows(f))
for f in OUT.rglob('requests.jsonl'):
    group='evaluation' if 'episodes' in f.relative_to(OUT).parts else str(f.parent.relative_to(OUT))
    c=costs.setdefault(group,{'requests':0,'input_tokens':0,'http_status':collections.Counter()})
    for r in rows(f):
        c['requests']+=1; c['http_status'][str(r['status'])]+=1
        tokens=r.get('response',{}).get('usage',{}).get('input_tokens',0); c['input_tokens']+=tokens
        if group!='evaluation': continue
        full[str(r['status'])]+=1
        if 'error_type' in r: full[r['error_type']]+=1
        state=decode(json.loads(r['request']['state']))
        assert hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()==r['raw_state_sha256']
        audit(state); audit(r['request']['questions']); exact+=1
        guard=len(r['request']['state'].encode())+len(json.dumps(r['request']['questions'],ensure_ascii=False).encode())+512
        assert guard<=24000; max_guard=max(max_guard,guard)
        api_seconds+=r['seconds']; max_tokens=max(max_tokens,tokens)
        if r['status']==200: model_ids.add(r['response']['model'])
assert model_ids=={'jev-1.13.0'}
for c in costs.values(): c['estimated_usd']=c['input_tokens']*.042/1e6
partitions=[r for f in (OUT/'episodes').rglob('partition_decisions.jsonl') for r in rows(f)]
statuses=collections.Counter(r['status'] for r in episodes)
outcomes=collections.Counter(outcome(r) for r in episodes)
success=outcomes['success']; wrong=outcomes['wrong_diagnosis']
type_acc=sum(r['metrics']['predicted_type']==r['metrics']['ground_truth_type'] for r in episodes)/100
loc_acc=sum(r['metrics']['location_correct'] for r in episodes)/100
weighted=sum(r['metrics']['network_size']*r['metrics']['diagnosis_success'] for r in episodes)/sum(r['metrics']['network_size'] for r in episodes)
assert abs(weighted-official['diagnosis_success_rate'])<1e-12
old39=[r for r in episodes if old[r['snapshot_id']]['status']=='Jev HTTP 400']
assert len(old39)==39
old39_outcomes=collections.Counter(outcome(r) for r in old39)
paired_old=collections.Counter(('both' if a and b else 'v3_only' if a else 'v2_only' if b else 'neither') for r in episodes for a,b in [(r['metrics']['diagnosis_success'],old[r['snapshot_id']]['metrics']['diagnosis_success'])])
paired_claude=collections.Counter(('both' if a and b else 'jev_only' if a else 'claude_only' if b else 'neither') for r in episodes for a,b in [(r['metrics']['diagnosis_success'],claude[r['snapshot_id']]['pass'])])
total_cost=sum(c['estimated_usd'] for c in costs.values())
context_errors=sum(r['status'] in ('Jev HTTP 400','error_ContextBudgetError') for r in episodes)
verification={'status':'completed_verified','episodes':100,'fingerprints_matching':100,'source_hashes_match':True,
 'request_roundtrips_exact':exact,'forbidden_private_keys':0,'model_ids':sorted(model_ids),
 'statuses':dict(statuses),'outcomes':dict(outcomes),'context_error_episodes':context_errors,
 'type_accuracy_unweighted':type_acc,'location_accuracy_unweighted':loc_acc,
 'original_39_outcomes':dict(old39_outcomes),'paired_v2':dict(paired_old),'paired_claude':dict(paired_claude),
 'costs':costs,'estimated_total_v3_usd':total_cost,'evaluation_http':dict(full),
 'partitioned_decisions':len(partitions),'partition_requests':sum(p['partitions'] for p in partitions),
 'maximum_partitions':max((p['partitions'] for p in partitions),default=1),
 'max_guard_bytes':max_guard,'largest_reported_input_tokens':max_tokens,'api_seconds_sum':api_seconds,
 'mean_episode_seconds':statistics.mean(r['seconds'] for r in episodes),
 'verified_at':datetime.now(timezone(timedelta(hours=9))).isoformat()}
save(OUT/'verification.json',verification)
by_type=[]
for ft in sorted({r['metrics']['ground_truth_type'] for r in episodes}):
    g=[r for r in episodes if r['metrics']['ground_truth_type']==ft]
    by_type.append(f"| {ft} | {len(g)} | {sum(outcome(r)=='success' for r in g)} | {sum(outcome(r)=='wrong_diagnosis' for r in g)} | {sum(r['status']!='completed' for r in g)} |")
type_table='\n'.join(by_type)
cost_table='\n'.join(f"| {name} | {c['requests']} | {c['input_tokens']:,} | ${c['estimated_usd']:.6f} | {dict(c['http_status'])} |" for name,c in costs.items())
cs=ref['summary']['episodes']; os=old_summary['official_summary']
report=f'''# Jev 문맥 한도 수정 후 공식 사례 재측정

작성일: 2026-10-11 KST. 검증 완료 시각: {verification['verified_at']}.

## 실행 결과

전체 100개 재측정과 원시 요청 검증을 완료했다. v2에서 39개였던 입력 한도 관련 미완료는 v3에서 {context_errors}개다. 성공 {success}개, 진단 제출 후 오진 {wrong}개, 기타 미완료 {100-success-wrong}개다. 모든 100개를 분모에 포함했다.

| 지표 | Jev v2 | Jev v3 | 원본 Claude Sonnet 4.5 |
|---|---:|---:|---:|
| 성공 사례 | 34/100 | {success}/100 | 66/100 |
| 공식 장비 수 가중 성공률 | {os['diagnosis_success_rate']:.2%} | {weighted:.2%} | {cs['diagnosis_success_rate']:.2%} |
| 유형 macro-F1 | {os['fault_type_macro_f1']:.4f} | {official['fault_type_macro_f1']:.4f} | {cs['fault_type_macro_f1']:.4f} |
| 공식 가중 위치 정확도 | {os['location_accuracy']:.2%} | {official['location_accuracy']:.2%} | {cs['location_accuracy']:.2%} |
| 공식 가중 평균 단계 | {os['avg_steps']:.2f} | {official['avg_steps']:.2f} | {cs['avg_steps']:.2f} |

v3 단순 유형 정확도 {type_acc:.2%}, 단순 위치 정확도 {loc_acc:.2%}. 성공은 유형과 위치가 모두 맞은 경우다. 위치 질문은 모델이 예측한 유형을 사용했다. 정답 유형을 제공한 보조 평가는 하지 않았다.

최종 실행 상태: `{dict(statuses)}`. 입력·실행 오류를 진단 오답과 별도로 분류한다. 원본 evaluator에 미완료 trace도 넣어 전체 100개를 집계한다.

| 실제 장애 유형 | 사례 수 | 성공 | 오진 | 미완료 |
|---|---:|---:|---:|---:|
{type_table}

기존 문맥 실패 39개의 재측정 결과: `{dict(old39_outcomes)}`. 이 부분집합 점수를 전체 성능으로 대신하지 않는다.

사례별 v2 비교: `{dict(paired_old)}`. v3_only는 새로 성공, v2_only는 이전에는 성공했지만 이번에는 실패한 경우다. Claude 비교: `{dict(paired_claude)}`. 단일 실행 결과이며 차이의 통계적 확실성을 주장하지 않는다.

## 수정과 검증

[실행 전 명세](../specs/netheal_official_jev_execution_v3_20261011.md)에 입력 규칙과 판정을 고정했다. 반복 자료, 배열, 과거 상태를 손실 없이 압축하고, 긴 이력은 동일 선택지로 나누어 평가한다. 분할 시 사건 수 가중 확률 평균으로 선택한다. 관측 절단·전문가 요약·정답 기반 후보 선별은 하지 않았다.

- 기존 요청 1,571개의 원자료와 디코딩 결과가 일치했다. 본 평가 실제 요청 {exact}개도 원자료 해시와 정확히 일치했다.
- 원본 Claude 100개와 시나리오 fingerprint 100/100 일치. 평가 소스와 압축 소스가 사전 동결 SHA-256과 일치한다.
- 본 평가 복원 입력과 질문에서 금지된 비공개 정답 키 0개. 모델 ID는 jev-1.13.0이다. 키 이름 검사만이 비누출의 완전한 증명은 아니며, 원본 공개 도구 경로와 입력 구성도 함께 확인했다.
- 실제 본 평가 분할 판단 {len(partitions)}회, 해당 묶음 요청 {sum(p['partitions'] for p in partitions)}회, 한 판단 최대 {verification['maximum_partitions']}묶음. 전체 요청 상태 `{dict(full)}`.
- 최대 요청 가드 {max_guard:,}바이트, 제공사가 보고한 최대 입력 {max_tokens:,}토큰. 가드는 실제 토크나이저 한도의 일반적 증명이 아니다.

## 해석과 한계

문맥 처리 결함과 진단 오답을 분리해서 해석해야 한다. 입력 실패 감소는 이 고정 평가에서 어댑터의 수용 가능성 개선을 지지한다. 관측 정보가 보존되어도 압축 문법 이해와 묶음 간 추론은 원래 입력과 다를 수 있다. 그러므로 v2 대비 점수 변화는 입력 표현·문맥 처리 변경을 포함한 전체 에이전트 결과다. 분할 없이 원시 전체 관측을 한 번에 읽는 모델 능력과 같다고 주장하지 않는다.

원본 환경·도구·채점 소스와 스냅샷을 사용했지만, 원본 Docker Linux Python 3.11.14의 전체 orchestration 대신 Windows Python 3.13.2와 직접 FastMCP handler를 사용했다. Jev는 선택형 도구/인자 인터페이스를 쓰며 Claude와 출력 방식이 다르다. 이미 관측한 평가셋의 입력 실패를 수정한 재측정이므로 새 미관측 평가가 아니다. 본 평가 도중 점수에 따라 설정을 바꾸지 않았다. 선택지 순서를 반복 바꾸는 추가 실험은 이번 재측정에 포함하지 않았다.

LAYA 학습, 8비트 optimizer, LoRA, Sol/Claude 추가 호출, 외부 리더보드 제출은 미실행이다. 이 결과로 LAYA의 성능을 판단할 수 없다. GPU는 로컬 모델 학습·추론에 사용하지 않았다.

## 시간과 비용

평균 에피소드 {verification['mean_episode_seconds']:.2f}초. 본 평가 API 대기 시간 합계 {api_seconds:.2f}초는 동시 요청을 합산한 값이다. 요청 속도 제한과 로컬 압축·실행 시간을 포함하므로 Claude 모델 자체 속도와 직접 비교하지 않는다.

v3 개발 확인·실패 요청·smoke·본 평가의 usage 기반 추정 비용은 **${total_cost:.6f}**다. v1/v2 비용은 이 금액에 포함하지 않았다.

| 단계 | 기록 요청 | 입력 토큰 | 추정 비용 | HTTP 상태 |
|---|---:|---:|---:|---|
{cost_table}

단가는 입력 100만 토큰당 $0.042, 출력 무료라는 [TypeSafe 모델 문서](https://docs.typesafe.ai/models)의 기존 확인값을 사용했다. 청구 확정액이 아니며 usage가 없는 실패/미기록 응답은 미계산일 수 있다. 개발 단계에서는 초기 압축만으로 100단계 입력이 여전히 400을 받아 분할을 추가했고, 그 기록도 보존했다.

## 재현과 산출물

```powershell
.runtime/jev-official-venv/Scripts/python.exe -X utf8 netheal-topology-generalization/scripts/jev_official_eval_v3.py evaluate
.runtime/jev-official-venv/Scripts/python.exe -X utf8 netheal-topology-generalization/scripts/report_jev_official_eval_v3.py
```

평가 명령은 완료된 결과를 재사용한다. 새로운 유료 재현 실행은 별도 OUT 디렉터리와 새 동결 명세가 필요하다. 기존 결과를 삭제하지 않는다.

원시 결과: `netheal-topology-generalization/results/netheal_official_jev_20261011_v3/summary.json`, `verification.json`, `episodes/*/attempt_*/requests.jsonl`, `tools.jsonl`, `helpers.jsonl`, 분할이 발생한 경우 `partition_decisions.jsonl`. 사전 명세·실행 소스·압축 소스의 사본과 SHA-256은 같은 결과 폴더에 있다. 원본 provenance와 스냅샷은 v1/upstream에 보존했다.

원본 비교 실행: [NetHeal 공개 리더보드](https://agentbeats.dev/manikyabard/netheal-ai-agent-benchmark), Run ID `685bad60-d554-4300-9c7e-e849301d6df7`, [원본 저장소](https://github.com/manikyabard/netheal-agentbeats-leaderboard), commit `496a07b13bdc1174325afefd8a7d63416e246c41`.
'''
(ROOT/'docs/results/netheal_official_jev_context_fix_20261011.md').write_text(report,encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False,indent=2))
