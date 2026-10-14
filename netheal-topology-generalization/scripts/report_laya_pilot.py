"""Verify and report completed LAYA runs; refuses to label partial runs complete."""
import json,math,hashlib
from pathlib import Path
from collections import Counter
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent
RUNS={"기본 LAYA":"laya_baseline_compact_fp32_v2","Head 학습 LAYA":"laya_head_compact_fp32_v2"}
def read(p):return json.loads(p.read_text(encoding="utf8"))
def lines(p):return [json.loads(l) for l in p.read_text(encoding="utf8").splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wilson(k,n):
    z=1.95996398454;p=k/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,mid-half),min(1,mid+half)]

def main():
    results={};raw={};verification={}
    train=read(PROJECT/'results/laya_head_compact_48ep_v2/protocol.json')
    ts=read(PROJECT/'results/laya_head_compact_48ep_v2/summary.json')
    for label,folder in RUNS.items():
        p=PROJECT/'results'/folder;m=read(p/'manifest.json');rows=lines(p/'episodes.jsonl');reqs=lines(p/'requests.jsonl')
        assert m['status']=='completed' and m['completed_episodes']==len(rows)==64
        assert len({r['case_id'] for r in rows})==64
        assert not ({r['episode_seed'] for r in rows}&set(train['selected_episode_seeds']))
        archive=p/'source_snapshot';archive.mkdir(exist_ok=True)
        for name,digest in m['source_sha256'].items():
            assert Path(name).name==name
            saved=archive/name
            if not saved.exists():
                original=PROJECT/'src'/name
                assert sha(original)==digest,name
                saved.write_bytes(original.read_bytes())
            assert sha(saved)==digest,name
        checkpoint=Path(m['checkpoint'])
        if not checkpoint.is_absolute():checkpoint=PROJECT/checkpoint
        assert sha(checkpoint/'model.safetensors')==m['checkpoint_sha256']
        assert all(not r['usage'].get('truncated') for r in reqs)
        assert all(r['steps']<=40 for r in rows)
        s=read(p/'summary.json');s['exact_ci95']=wilson(sum(r['exact'] for r in rows),len(rows))
        s['question_latency_p50_p95']=np.percentile([r['seconds'] for r in reqs],[50,95]).tolist()
        s['peak_gpu_allocated_bytes']=m['peak_gpu_allocated_bytes'];s['cpu_fallback_count']=m['cpu_fallback_count']
        s['question_count']=len(reqs)
        token_counts=[r['usage']['input_tokens'] for r in reqs]
        s['question_input_tokens_p50_p95_max']=[*np.percentile(token_counts,[50,95]).tolist(),max(token_counts)]
        s['questions_over_checkpoint_default_512']=sum(t>512 for t in token_counts)
        precision=[];recall=[]
        for ft in sorted({r['fault_type'] for r in rows}):
            tp=sum(r['fault_type']==ft and r['predicted_type']==ft for r in rows)
            predictions=sum(r['predicted_type']==ft for r in rows);truth=sum(r['fault_type']==ft for r in rows)
            precision.append(tp/predictions if predictions else 0);recall.append(tp/truth)
        s['fault_macro_precision']=float(np.mean(precision));s['fault_macro_recall']=float(np.mean(recall))
        successful=[r['wall_seconds'] for r in rows if r['exact']]
        s['mean_seconds_successful_only']=float(np.mean(successful)) if successful else None
        s['faults']={ft:dict(n=len(rs),exact=sum(r['exact'] for r in rs)/len(rs))
            for ft in sorted({r['fault_type'] for r in rows}) if (rs:=[r for r in rows if r['fault_type']==ft])}
        s['actions']=dict(Counter(t['action']['kind'] for r in rows for t in r['trace']))
        results[label]=s;raw[label]={r['case_id']:r for r in rows}
        verification[label]=dict(episodes=64,no_train_overlap=True,no_truncation=True,source_hashes_match=True,
            episode_sha256=sha(p/'episodes.jsonl'),requests_sha256=sha(p/'requests.jsonl'))
    a,b=list(raw.values());assert a.keys()==b.keys()
    for k in a:assert all(a[k][v]==b[k][v] for v in ('episode_seed','fault_type','fault_location'))
    api={r['case_id']:r for r in lines(PROJECT/'results/api_pilot_20261009/episodes.jsonl') if r['provider']=='gpt'}
    assert api.keys()==a.keys()
    for k in a:assert all(a[k][v]==api[k][v] for v in ('episode_seed','fault_type','fault_location'))
    verification['matches_existing_api_pilot_cases']=True
    keys=sorted(a);delta=np.array([int(b[k]['exact'])-int(a[k]['exact']) for k in keys])
    rng=np.random.default_rng(20261009)
    groups=[[i for i,k in enumerate(keys) if a[k]['condition']==c] for c in sorted({r['condition'] for r in a.values()})]
    boot=np.zeros(10000)
    for ids in groups:boot+=delta[rng.choice(ids,(10000,len(ids)),replace=True)].sum(axis=1)/len(keys)
    improved=int(sum(delta==1));worsened=int(sum(delta==-1));discordant=improved+worsened
    pval=min(1.,2*sum(math.comb(discordant,i) for i in range(min(improved,worsened)+1))/2**discordant) if discordant else 1.
    paired=dict(delta=float(delta.mean()),conditional_bootstrap_ci95=np.percentile(boot,[2.5,97.5]).tolist(),
        improved=improved,worsened=worsened,mcnemar_exact_p=pval,
        warning='Bootstrap interval is conditional on these pilot cases. A degenerate all-zero interval does not prove equivalence.')
    controlled=None
    control_path=PROJECT/'results/laya_diagnosis_only_v2'
    if (control_path/'manifest.json').exists() and read(control_path/'manifest.json')['status']=='completed':
        controlled=read(control_path/'summary.json');preds=lines(control_path/'predictions.jsonl')
        assert len(preds)==576 and all(not u.get('truncated') for r in preds for u in r['usage'])
        assert not ({r['episode_seed'] for r in preds}&set(train['selected_episode_seeds']))
        for label in ('base','head'):
            rs=[r for r in preds if r['model']==label]
            assert len(rs)==len({r['episode_seed'] for r in rs})==288
            controlled[label]['exact_ci95']=wilson(sum(r['exact'] for r in rs),len(rs))
            controlled[label]['predicted_type_counts']=dict(Counter(r['predicted_type'] for r in rs))
        ca={r['episode_seed']:r for r in preds if r['model']=='base'}
        cb=[r for r in preds if r['model']=='head']
        controlled['paired']=dict(
            changed_locations=sum(r['predicted_location']!=ca[r['episode_seed']]['predicted_location'] for r in cb),
            improved=sum(r['exact'] and not ca[r['episode_seed']]['exact'] for r in cb),
            worsened=sum(not r['exact'] and ca[r['episode_seed']]['exact'] for r in cb))
    report=dict(status='completed_verified',runs=results,paired=paired,verification=verification,training=ts,
        diagnosis_only=controlled)
    out=PROJECT/'results/laya_pilot_report_v2';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    text=['# LAYA 소규모 학습·실제 진단 pilot','',
      '2026-10-09. 기본 모델과 head만 학습한 모델의 **64개 동일 사례씩, 총 128개 rollout**을 완료했다. 추가 GPT/Jev API 호출은 없었다.',
      '', '## 어떤 가설을 검증했나','',
      '- L1: 공개 사전학습 LAYA가 관측만으로 도구를 고르고 진단까지 끝낼 수 있는가? 실제 환경에서 40회 행동 제한으로 평가했다.',
      '- L2의 제한된 점검: encoder를 고정한 소규모 head 학습이 같은 모델의 진단 성공률을 높이는가? 전체 encoder 미세조정의 효과를 검증한 것은 아니다.',
      '- L3: 학습 범위(Star/Linear, 크기 4–6) 밖의 크기·Mesh·Random 성능은 어떠한가? 조건별 pilot 수치는 기록했지만 일반화 효과를 확정할 표본은 아니다.',
      '', '## 설정과 완료 범위','',
      '- laya 0.4.1, 공개 checkpoint revision `7b928d828b7b0e022f929d9bd2e44165aa270148`.',
      '- GPU는 GTX 1660 SUPER 6GB이며, CPU 전용 PyTorch가 원인이었던 초기 인식 오류를 수정했다. 별도 LAYA 환경에 torch 2.12.1+cu126을 설치했다.',
      '- 양쪽 추론은 FP32, autocast 끔, 동시 실행 1이다. GTX 16xx에서 느렸던 기본 FP16 점검은 미완료 기록으로 따로 남겼다.',
      '- 같은 공개 그래프 특징을 짧은 표로 전달하고 행동·대상·장애 유형 질문을 분리했다. 후속 질문은 정답이 아닌 모델의 앞선 답변으로 고른다.',
      '- 학습/보정/검증은 물리 에피소드 단위로 분리했다. 이번 학습은 train의 24개 조합에서 2개씩 48개 에피소드, 516개 질문, 2 epochs, seed 41이다.',
      '- 전문가 데모는 기존 v3의 성공한 전문가 궤적에서 가져왔다. 마지막 상태 진단 보조 평가 역시 이 성공 데모 분할이며, 실제 64개 pilot은 별도의 새 seed 사례를 성공 여부와 무관하게 모두 평가한다.',
      f'- Head 학습 시간 {ts["seconds"]:.2f}초, 최대 PyTorch GPU 할당 {ts["peak_gpu_allocated_bytes"]/2**30:.2f}GiB. Encoder 고정; confidence 보정은 수행하지 않았다.',
      '- 전체 encoder 학습은 8개 질문의 실행 점검만 통과했다. 81.41초, 최대 할당 8.35GB로 물리 VRAM을 넘었으므로 이 결과를 본 미세조정 완료로 부르지 않는다.',
      '', '## 실제 장애 탐색 결과','',
      '| 지표 | 기본 LAYA | Head 학습 LAYA |','|---|---:|---:|']
    vals=list(results.values())
    metrics=[('장애 유형+위치 정확도','exact','pct'),('정확도 95% Wilson 구간','exact_ci95','ci'),
      ('장애 유형 정확도','type_accuracy','pct'),('위치 정확도','location_accuracy','pct'),('장애 유형 macro-F1','fault_macro_f1','num'),
      ('장애 유형 macro-precision','fault_macro_precision','num'),('장애 유형 macro-recall','fault_macro_recall','num'),
      ('전체 사례 중 오진 비율','wrong_diagnosis_rate','pct'),('평균 완료/종료 시간(초)','mean_seconds','num'),
      ('p95 완료/종료 시간(초)','p95_seconds','num'),('평균 도구 호출','mean_tool_calls','num'),
      ('도구 호출 중 반복 비율','repeated_tool_rate','pct'),('평균 가상 도구 비용','mean_tool_cost','num'),
      ('LAYA 질문 수','question_count','int'),('CPU fallback 수','cpu_fallback_count','int')]
    for title,key,fmt in metrics:
        def show(v):
            if fmt=='pct':return f'{v*100:.1f}%'
            if fmt=='ci':return f'{v[0]*100:.1f}–{v[1]*100:.1f}%'
            if fmt=='int':return str(v)
            return f'{v:.3f}'
        text.append('| '+title+' | '+' | '.join(show(s[key]) for s in vals)+' |')
    text+=['','종료 사유: '+ '; '.join(f'{k} `{s["termination_counts"]}`' for k,s in results.items())+'.','',
        '두 모델의 LAYA API 비용은 $0이다. 전력·장비 감가상각 비용은 측정하지 않았다. 가상 도구 비용은 시뮬레이터 점수이며 달러 과금이 아니다.',
        '', '| 조건 | 사례 수 | 기본 Exact | Head 학습 Exact |','|---|---:|---:|---:|']
    for c in vals[0]['conditions']:
        text.append(f'| {c} | {vals[0]["conditions"][c]["n"]} | {vals[0]["conditions"][c]["exact"]*100:.1f}% | {vals[1]["conditions"][c]["exact"]*100:.1f}% |')
    if controlled:
        text+=['','## 증거 수집과 증거 해석을 분리한 점검','',
            '학습에 쓰지 않은 validation의 288개 에피소드에서, 전문가가 공개 도구로 증거를 수집한 마지막 상태만 제공했다. 행동 선택과 종료 판단은 생략하고, 장애 유형 다음에 **모델이 예측한 유형에 맞는** 위치 질문을 했다. 정답 유형으로 위치 질문을 고르지 않았다.',
            '', '| 지표 | 기본 LAYA | Head 학습 LAYA |','|---|---:|---:|']
        for title,key in [('유형+위치 정확도','exact'),('유형 정확도','type_accuracy'),('위치 정확도','location_accuracy')]:
            text.append('| '+title+' | '+' | '.join(f'{controlled[k][key]*100:.1f}%' for k in ('base','head'))+' |')
        text+=['', '이 평가는 64개 탐색 사례와 다른 분할이며, 전문가가 성공한 Star/Linear 크기 4–6 사례에 한정된다. 수치 차이가 탐색·종료 판단 문제를 살펴볼 단서는 되지만, 그 요인만이 원인이라는 인과 증명은 아니다. 최종 진단 정확도로 탐색 정확도를 대체해서는 안 된다.']
        text+=['', '예측 유형 분포: '+ '; '.join(f'{k}: `{controlled[k]["predicted_type_counts"]}`' for k in ('base','head'))+'.',
            f'학습 후 위치 예측이 달라진 사례는 {controlled["paired"]["changed_locations"]}개이며, 정답으로 개선 {controlled["paired"]["improved"]}개·오답으로 악화 {controlled["paired"]["worsened"]}개였다. 네 장애 유형이 균등한 평가에서 한 유형만 계속 고르면 유형 정확도 25%가 나오므로, 그 수치 자체를 증거 해석 성공으로 볼 수 없다.']
    if all(s['exact']==0 for s in vals):
        text+=['', '**현재 구성은 실제 장애 탐색에 사용할 수준을 보여주지 못했다. 소규모 head 학습의 성공률 개선도 관측하지 못했다.** 실패는 학습 미실행이 아니라 실제 출력 부분의 갱신 이후에도 남았다. 단, 입력 표현과 학습 규모를 분리한 비교가 없으므로 프롬프트 문제인지 LAYA의 능력 한계인지 확정할 수 없다.']
    text+=['','## 결론과 한계','',
      f'- 같은 사례에서 학습 후 개선 {improved}개, 악화 {worsened}개; Exact 변화 {paired["delta"]*100:+.1f}%p. McNemar 양측 exact p={pval:.4f}.',
      '- 학습 손실 감소만으로 진단 성능 향상을 주장하지 않는다. 기본·학습 모델의 실제 성공률과 반복 도구 호출을 함께 판단해야 한다.',
      '- 오진율이 낮아도 진단 제출 자체가 적고 행동 제한 종료가 많으면 좋은 결과가 아니다. 유형 macro-F1 계산에서도 미제출은 정답 유형의 false negative로 포함했다.',
      '- 이 결과는 소규모 head 학습과 특정 입력 형식에 대한 결과다. LAYA 자체의 성능 상한이나 충분한 전체 미세조정의 결과로 일반화할 수 없다.',
      '- 원래 checkpoint 설정은 max_len 512, head_max_len 192이다. 정보 누락을 막기 위해 이번에는 4096/1024로 확대했고 매 호출에서 잘림을 검사했다. 처리 가능한 길이와 그 길이에서의 성능 보장은 다르며, 숫자 표 입력과 긴 문맥에 대한 적응 부족을 배제하지 못한다.',
      '- GPT/Jev pilot과 물리 사례·공개 증거·행동 제한은 같다. 그러나 질문 형식, 추가 학습 여부, 로컬 GPU와 원격 API 실행 환경이 달라 모델 자체의 공정한 최종 순위라고 부를 수 없다.',
      '- 정상 상태 오경보, 복합 장애, 실장비 검증, ID 변경·선택지 순서 실험, 여러 학습 seed 반복, 보정된 확률은 아직 검증하지 않았다.',
      '- 이 실험은 장애 탐색 정책을 평가한다. 실제 네트워크의 처리량 개선·패킷 손실 감소·복구 시간 단축은 측정하지 않았으며, 관측된 ping 지연과 진단 프로그램의 실행 시간을 혼동하지 않는다.',
      '- 64개 독립 사례는 pilot이다. 동일 사례의 수천 번 질문을 독립 표본 수로 세지 않는다. 결과를 본 뒤 개발에 재사용한 이 사례는 최종 미공개 평가셋으로 사용할 수 없다.',
      '- 실행 순서는 기본 모델 다음 학습 모델이며 순서 교차 반복은 하지 않았다. 지연은 이 장비에서의 관측치이며 순수한 모델 구조의 속도 차이를 추정한 결과가 아니다.',
      '- 다음 단계는 학습용 검증 분할에서 더 충분한 적응 학습과 입력·질문 ablation을 선택한 뒤, 별도 새 사례에서 고정된 프로토콜로 확인하는 것이다. 유료 API 재평가는 그 이후에 수행한다.',
      '', '## 산출물','',
      '- `netheal-topology-generalization/results/laya_pilot_report_v2/summary.json`: 집계·paired 비교·검증 결과.',
      '- `results/laya_baseline_compact_fp32_v2/`, `results/laya_head_compact_fp32_v2/`: 질문 로그, 실제 행동/도구 결과, 사례별 결과, 해시 manifest, `source_snapshot/`의 실행 소스 사본.',
      '- `results/laya_head_compact_48ep_v2/`: 고정 학습 계획, 결과, 저장 checkpoint.',
      '- [실험 설계](../specs/laya_diagnosis_v1.md).']
    (ROOT/'docs/results/laya_pilot_v2.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps({'status':'completed_verified','paired':paired},ensure_ascii=False))
if __name__=='__main__':main()
