"""Verify completed fixed runs and report all outcomes, without model selection."""
import json, statistics, time, shutil
from pathlib import Path
from collections import Counter
import numpy as np
from laya_observation_data import OUT, PROJECT, dump, sha, digest
from laya_observation_controls import RUNS, metrics, record

def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    frozen=read(OUT/'training_complete.json');opened=read(OUT/'evaluation_opened.json')
    assert opened['opened_unix']>frozen['completed_unix']
    assert opened['frozen_training_manifest_sha256']==sha(OUT/'training_complete.json')
    restoration=read(OUT/'restoration_audit.json');assert len(restoration['checks'])==6
    batch_audit=read(OUT/'batch_conformance.json');assert batch_audit['status']=='completed_verified'
    data=read(OUT/'data_audit.json');extended=read(OUT/'extended_data_audit.json')
    assert extended['status']=='completed_verified'
    for name,h in data['data_sha256'].items():assert sha(OUT/name)==h
    cases={s:[json.loads(l) for l in (OUT/f'{s}_cases.jsonl').read_text(encoding='utf8').splitlines()] for s in ['train','test']}
    def pattern(r):
        return digest([[kind,name,[k for k in keys if k not in ['observed_latency_div_100ms','step_fraction']],
                        [v for k,v in zip(keys,values) if k not in ['observed_latency_div_100ms','step_fraction']]]
                       for kind,name,keys,values in r['facts']])
    train_patterns={pattern(r) for r in cases['train']}
    pattern_matches=sum(pattern(r) in train_patterns for r in cases['test'])
    runs=[];training=[]
    for arm,seed in [('table','base'),('prose','base')]+RUNS:
        folder=OUT/f'{arm}_{seed}';r=read(folder/'evaluation.json')
        assert r['status']=='completed'
        for split in ['train','test']:
            assert metrics(r[split]['records'])==r[split]['metrics']
            assert len(r[split]['records'])==len(cases[split])
            for rec,raw,case in zip(r[split]['records'],r[split]['raw'],cases[split]):
                assert rec['episode_seed']==raw['episode_seed']==case['episode_seed']
                q=case['questions'];preds=raw['canonical']
                for name,p in preds.items():
                    keys=list(q[name]['criteria']);assert p['prediction']==keys[p['order'][int(np.argmax(p['logits']))]]
                ft=preds['fault']['prediction'];branch='diagnosis_node' if ft in ['device_failure','misconfiguration'] else 'diagnosis_edge'
                assert rec['predicted_type']==ft and rec['predicted_location']==preds[branch]['prediction']
        assert len(r['test']['type_orders'])==24 and len(r['test']['location_orders'])==3
        assert r['test']['batched_canonical_comparisons']==192
        assert r['test']['batched_canonical_max_logit_difference']<1e-4
        for j,summary in enumerate(r['test']['type_orders']):
            recs=[]
            for raw,case in zip(r['test']['raw'],cases['test']):
                p=raw['type_orders'][j];keys=list(case['questions']['fault']['criteria'])
                assert p['order']==summary['order']
                assert p['prediction']==keys[p['order'][int(np.argmax(p['logits']))]]
                ft=p['prediction'];branch='diagnosis_node' if ft in ['device_failure','misconfiguration'] else 'diagnosis_edge'
                oracle_branch='diagnosis_node' if case['expected']['fault'] in ['device_failure','misconfiguration'] else 'diagnosis_edge'
                recs.append(record(case,ft,raw['canonical'][branch]['prediction'],raw['canonical'][oracle_branch]['prediction']))
            assert metrics(recs)==summary['metrics']
        for j,summary in enumerate(r['test']['location_orders']):
            recs=[]
            for raw,case in zip(r['test']['raw'],cases['test']):
                ft=raw['canonical']['fault']['prediction'];branch='diagnosis_node' if ft in ['device_failure','misconfiguration'] else 'diagnosis_edge'
                oracle_branch='diagnosis_node' if case['expected']['fault'] in ['device_failure','misconfiguration'] else 'diagnosis_edge'
                for k in ['diagnosis_node','diagnosis_edge']:
                    p=raw['location_orders'][k][j];keys=list(case['questions'][k]['criteria'])
                    assert p['prediction']==keys[p['order'][int(np.argmax(p['logits']))]]
                recs.append(record(case,ft,raw['location_orders'][branch][j]['prediction'],raw['location_orders'][oracle_branch][j]['prediction']))
            assert metrics(recs)==summary['metrics']
        memorized=all(r['train']['metrics'][k]>=.95 for k in ['type_accuracy','exact'])
        passed=all(r['test']['metrics'][k]>=.95 for k in ['type_accuracy','macro_f1','location_accuracy','exact']) and all(o['metrics']['exact']>=.95 for kind in ['type_orders','location_orders'] for o in r['test'][kind])
        r['memorization_pass']=memorized;r['new_case_pass']=passed;runs.append(r)
        if seed!='base':
            t=read(folder/'training.json');assert t['optimizer_updates']==1024 and t['type_updates']==t['location_updates']==512
            assert t['checkpoint_sha256']==sha(folder/'trainable.pt')==frozen['checkpoints'][f'{arm}_{seed}']
            training.append(t)
    paired={};rng=np.random.default_rng(731)
    trained={a:[r for r in runs if r['arm']==a and r['seed']!='base'] for a in ['table','prose']}
    for metric,field in [('type_accuracy','type_correct'),('location_accuracy','location_correct'),('exact','exact')]:
        delta=np.array([[float(p['test']['records'][i][field])-float(t['test']['records'][i][field])
            for i in range(64)] for t,p in zip(trained['table'],trained['prose'])]).mean(0)
        estimates=[float(delta[rng.integers(0,64,64)].mean()) for _ in range(2000)]
        paired[metric]={'prose_minus_table':float(delta.mean()),'case_bootstrap_95':np.quantile(estimates,[.025,.975]).tolist()}
    # macro-F1 is nonlinear, so resample cases and recompute confusion matrices.
    differences=[]
    for _ in range(2000):
        ix=rng.integers(0,64,64)
        differences.append(statistics.mean(metrics([p['test']['records'][i] for i in ix])['macro_f1']-metrics([t['test']['records'][i] for i in ix])['macro_f1'] for t,p in zip(trained['table'],trained['prose'])))
    paired['macro_f1']={'prose_minus_table':statistics.mean(p['test']['metrics']['macro_f1']-t['test']['metrics']['macro_f1'] for t,p in zip(trained['table'],trained['prose'])), 'case_bootstrap_95':np.quantile(differences,[.025,.975]).tolist()}
    lines=['# LAYA 실제 공개 관측의 표·문장 비교','',
        '2026-10-10에 사전 계획을 고정하고 수행한 시뮬레이터 관측 실험이다. 공개 도구로 수집한 실제 장애 증거를 사용했으며, 정답에서 만든 합성 신호가 아니다. 실장비 네트워크 진단 성능이나 LAYA 도구 탐색 성능을 측정한 것은 아니다.','',
        '## 실행과 데이터','',
        '- 완료: 관측96개 생성, 표/문장 동등성 검증, 6개 모델 학습(각1024 optimizer updates), 기본 모델2표현 및 학습 모델6개의 학습/새 평가, 선택지 순서 검사, 6개 가중치 복원 검증.',
        '- 미실행: 전체 Encoder 미세조정, 새 LAYA 도구 탐색 rollout, 유료 GPT/Jev API, 정상/복합 장애·외부 토폴로지·실장비 평가.',
        '- 중단: 문장 seed63의 첫 실행은 세션 중단 뒤 프로세스가 사라졌고 완료 가중치가 없었다. 원래 폴더를 prose_63_interrupted_attempt1로 보존하고 같은 설정·기본 checkpoint에서 재시작했다. 완료된5개 실행은 재학습하지 않았다. 중단 실행의 정확한 갱신 수·시간은 복구할 수 없으며 아래 시간 합계에 포함되지 않는다.',
        '- Star/Linear 각4노드×4유형에서 학습4개/평가8개씩: 학습32·평가64. 학습91000000대, 평가92000000대의 별도 episode seed. 유형마다 평가16개다.',
        f'- 후보{data["attempts"]}개, 채택{data["accepted"]}개, 중복/제외{data["excluded"]}개. 기존 결과{len(extended["historical_files"])}파일·저장 compact 관측{extended["historical_compact_state_records"]}건과 seed 및 기록 가능한 관측을 검사했다. 기존 전체 LAYA 데이터2112에피소드의 모든 단계도 검사했다. 관측 원문이 없는 기록은 seed만 검사할 수 있었다.',
        f'- 학습/평가 관측의 완전일치와 step_fraction 제외 핵심일치 모두0건. 다만 지연 연속값과 step_fraction을 제거하면 새 평가{pattern_matches}/64개가 학습과 같은 구조적 관측 패턴이다. 새로운 물리 에피소드이며, 새로운 장애 패턴/토폴로지 일반화로 부르지 않는다.',
        '- scan → 모든 장치 status → 모든 장치 interfaces → 모든 명목 방향 링크 ping의 고정 순서로 수집했다. 정답을 보고 도구를 선택하거나 종료하지 않았다. 각 사례의 실제 도구 결과와 전체 공개 그래프 필드를 candidates.jsonl에 보존했다.',
        f'- 전문가 성공: 학습{data["expert_success_by_split"]["train"]}/32, 평가{data["expert_success_by_split"]["test"]}/64. 전문가 성공으로 선별하지 않았다. 전문가 판단은 raw memory를 이용하므로 압축 그래프만으로 동일한 성능이 보장되는 것은 아니다.',
        '- 표와 영문 문장은 같은 GraphObs 전체 필드·ID·순서·5자리 유효숫자 값을 전달한다. 양방향 파서로192개 표현의 필드/값 일치를 확인했다. 정답·전문가 결론·사례ID는 입력에 없다. 원시 도구 정보 전체와 GraphObs가 동등하다는 주장은 아니다.',
        '- 최종6개 가중치와 해시를 모두 저장한 뒤 새 평가를 개방했다. 중간 점수로 조건을 고르거나 재학습하지 않았다.','',
        '## 조건과 사전 기준','',
        '기본 checkpoint revision 7b928d828b7b0e022f929d9bd2e44165aa270148, FP32, 마지막 Encoder 블록+Head만 학습했다. Encoder/Head LR=2.5e-5/1e-4, AdamW soft-CE, microbatch1/accumulation1, 선택지 섞기, seed41/52/63이다. 32사례×2질문×16epochs=1024 실제 optimizer.step 호출을 계수했다. 유형·위치 각각512회이므로 기존 유형 전용1024회 실험과 유형 학습 노출량은 다르다. cosine schedule은1024회에 맞췄다.', '',
        '암기 통과는 학습 유형·Exact 모두95%. 새 사례 통과는 유형 정확도·macro-F1·위치·Exact 모두95%이고, 유형24순서 및 위치3순서의 Exact도 모두95% 이상이다. 최종 checkpoint를 사용했다. 기준 미달은 모델 고유 능력의 불가능성 판정이 아니다.','',
        '## 학습 사례와 새 사례 결과','',
        '| 표현 | seed | 학습 유형 | 학습 Exact | 새 유형 | 새 macro-F1 | 새 위치 | 새 Exact | 정답유형 위치(보조) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        a=r['train']['metrics'];b=r['test']['metrics']
        lines.append(f'| {r["arm"]} | {r["seed"]} | {a["type_accuracy"]:.2%} | {a["exact"]:.2%} | {b["type_accuracy"]:.2%} | {b["macro_f1"]:.4f} | {b["location_accuracy"]:.2%} | {b["exact"]:.2%} | {b["oracle_location_accuracy"]:.2%} |')
    lines+=['','주 결과의 위치 질문은 모델이 예측한 유형에 따라 선택했다. 정답 유형을 이용한 위치 점수는 마지막 열의 보조 평가에만 해당한다. 위치는 유형을 틀려도 위치 자체가 맞으면 정답으로 센다. 링크 역방향은 동등하게 처리한다. 학습 사례 점수도 예측 유형으로 분기한다.','',
        '## 선택지 순서 민감도','',
        '| 표현 | seed | 유형24순서 유형 평균/최저 | 유형24순서 Exact 평균/최저 | 위치3순서 Exact 최저 | 최대 예측 변경률 |',
        '|---|---|---:|---:|---:|---:|']
    for r in runs:
        tt=r['test']['type_orders'];ll=r['test']['location_orders']
        ta=[o['metrics']['type_accuracy'] for o in tt];ex=[o['metrics']['exact'] for o in tt]
        lines.append(f'| {r["arm"]} | {r["seed"]} | {statistics.mean(ta):.2%}/{min(ta):.2%} | {statistics.mean(ex):.2%}/{min(ex):.2%} | {min(o["metrics"]["exact"] for o in ll):.2%} | {max(o["prediction_disagreement"] for o in tt+ll):.2%} |')
    lines+=['','유형24순서에서는 위치 순서를 고정하고, 위치 canonical/reverse/rotate1에서는 유형 순서를 고정했다. 전체 순서 조합의 전수평가는 아니다. 동일64개 사례의 반복 측정이며 독립 표본을 늘려 계산하지 않았다.','',
        '## 표현 비교와 판정','']
    for arm,rr in trained.items():
        lines.append(f'- {arm}: 암기 기준 {sum(r["memorization_pass"] for r in rr)}/3, 새 사례 기준 {sum(r["new_case_pass"] for r in rr)}/3. 새 유형 평균 {statistics.mean(r["test"]["metrics"]["type_accuracy"] for r in rr):.2%}, Exact 평균 {statistics.mean(r["test"]["metrics"]["exact"] for r in rr):.2%}.')
    for k,v in paired.items():
        lo,hi=v['case_bootstrap_95'];lines.append(f'- 문장−표 {k}: {v["prose_minus_table"]*100:+.2f}%p, 사례 bootstrap95% 구간 [{lo*100:+.2f}, {hi*100:+.2f}]%p.')
    lines+=['','구간은 같은64개 사례에서 세 seed의 차이를 평균한 뒤 사례를 재표집했다. seed 모집단의 불확실성이나 다중 비교 보정은 포함하지 않는다. 표현은 정보량을 맞췄지만 토큰 수·문장 구조·계산량은 다르므로 순수한 자연어 의미 효과만 분리한 실험은 아니다.','']
    if not any(r['new_case_pass'] for r in runs if r['seed']!='base'):
        lines.append('- “이 고정 예산·표현·부분 학습으로 새 관측에서 안정적인95% 판단을 확보한다”는 사전 가설은 통과하지 못했다. 충분한 공개 관측을 수집해도 이번 판단 구성에서 실패가 남으므로, 이번 실패를 탐색 단계만으로 설명할 수 없다.')
    else:
        lines.append('- 새 사례 사전 기준을 통과한 실행이 있다. 통과 seed 수와 순서 최저값을 함께 보며, 이 작은 동일 분포 평가를 실장비·다른 토폴로지의 보장으로 확장하지 않는다.')
    if not any(r['memorization_pass'] for r in runs if r['seed']!='base'):
        lines.append('- 학습 사례 암기 기준도 모두 미달했다. 따라서 새로운 사례에서의 낮은 점수를 일반화 문제 하나로 해석할 수 없다. 최적화·노출 예산·표현 적응 문제는 아직 분리되지 않았다.')
    for arm,rr in trained.items():
        base=next(r for r in runs if r['arm']==arm and r['seed']=='base')
        d=statistics.mean(r['test']['metrics']['exact']-base['test']['metrics']['exact'] for r in rr)
        lines.append(f'- {arm}의 기본 모델 대비 새 Exact 평균 변화는 {d*100:+.2f}%p다. 손실 변화와 별개로 실제 판단 점수의 개선량을 보고한다.')
    lo,hi=paired['exact']['case_bootstrap_95']
    if lo<=0<=hi:
        lines.append('- 문장−표 Exact 차이 구간이0을 포함한다. 이 표본으로 문장 표현의 우위를 확정할 수 없으며, 두 표현이 동등하다고 입증한 것도 아니다.')
    elif lo>0:
        lines.append('- 문장−표 Exact 차이의 사례 bootstrap 구간이 양수여서 이 고정 조건의 문장 표현 개선을 지지한다. 입력 길이 차이와 세 seed의 제한 때문에 일반적인 자연어 우위를 확정하지 않는다.')
    else:
        lines.append('- 문장−표 Exact 차이의 사례 bootstrap 구간이 음수다. 이 조건에서 문장으로 바꾸면 개선된다는 가설을 지지하지 않는다. 다른 문장화 방식까지 배제하지 않는다.')
    lines+=['- 관측 원문을 이용한 전문가 성공은 이 시뮬레이터 사례에 진단 가능한 공개 증거가 있음을 지지한다. 그 정보가 현재 그래프 압축·표현에서 충분히 보존되고 LAYA가 활용할 수 있는지는 별도 문제다. 모델 자체의 능력 상한이나 전체 Encoder 학습의 효과는 이번 결과로 기각할 수 없다.','',
        '## 자원과 재현','',
        '| 표현 | seed | 학습 초 | 최대 allocated GiB | 최대 reserved GiB | GPU 평균 사용률 | 장치 최대 MiB | 입력 tokens 최소–최대 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for t in training:
        samples=[x for x in t['gpu_samples'] if 'utilization_percent' in x]
        lines.append(f'| {t["arm"]} | {t["seed"]} | {t["seconds"]:.1f} | {t["peak_allocated_bytes"]/2**30:.2f} | {t["peak_reserved_bytes"]/2**30:.2f} | {statistics.mean(x["utilization_percent"] for x in samples):.1f}% | {max(x["device_memory_MiB"] for x in samples):.0f} | {t["sequence_tokens"]["min"]}–{t["sequence_tokens"]["max"]} |')
    lines+=['','| 표현 | seed | 새 평가 전체 초(순서 포함) | canonical 3질문 평균/p95 초 | 평가 peak GiB |', '|---|---|---:|---:|---:|']
    for r in runs:
        e=r['test'];lines.append(f'| {r["arm"]} | {r["seed"]} | {e["seconds"]:.1f} | {e["canonical_three_question_mean_seconds"]:.3f}/{e["canonical_three_question_p95_seconds"]:.3f} | {r["peak_allocated_bytes"]/2**30:.2f} |')
    lines+=['','매 canonical 평가에서 유형·node·edge 세 질문을 계산해 oracle 보조 결과도 확보했다. 시간은 실제 운영시 두 질문만 계산하는 지연이 아니다. GPU peak는 PyTorch 할당량이며 nvidia-smi의 전체 장치 메모리·사용률 시계열도 JSON에 보존했다. 장치 메모리에는 디스플레이 등 다른 프로세스가 포함될 수 있다. API 비용0, 전력 비용 미측정.','',
        '모든 입력은4096/1024 예산에서 공식 tokenizer로 잘림·후보 누락/병합을 검사했다. 저장 가중치에는 마지막 Encoder 블록과 Head가 포함된다. 기본 모델에 다시 로드한6개 실행의 canonical logits 최대오차는 '+str(max(x['max_logit_error'] for x in restoration['checks']))+'이며 예측·점수가 일치했다.','',
        '새 평가 개방 전 실행 보완안에 따라 순서 반복 추론만 batch4로 묶었다. 학습 사례에서 두 표현·세 질문의 순서별 단일/배치 추론을 비교하고, 평가에서도 모델별192개 canonical 비교에서 logits 오차<1e-4와 예측 일치를 확인했다. canonical 추론은 단일 실행이다. 학습·데이터·판정 기준은 변경하지 않았다.','',
        '```powershell', '# netheal-topology-generalization 디렉터리에서 실행', '$env:PYTHONPATH = (Resolve-Path ../netheal-ai-agent-benchmark).Path', '../.runtime/venv/Scripts/python.exe scripts/laya_observation_data.py', '../.runtime/venv/Scripts/python.exe scripts/verify_laya_observation_data.py', '../.runtime/laya-venv/Scripts/python.exe scripts/laya_observation_controls.py train', '../.runtime/laya-venv/Scripts/python.exe scripts/laya_observation_batched_eval.py conformance', '../.runtime/laya-venv/Scripts/python.exe scripts/laya_observation_batched_eval.py evaluate', '../.runtime/laya-venv/Scripts/python.exe scripts/laya_observation_controls.py restore', '../.runtime/laya-venv/Scripts/python.exe scripts/report_laya_observation_controls.py','```','',
        '데이터 생성은 출력 폴더가 이미 존재하면 거부한다. 새 재현에는 스크립트의 OUT을 새 경로로 지정하고 사전 계획을 복사한다. 기존 환경·기존 결과·run_nonlink_single_fault_audit.py는 변경하지 않았다. 초기 샌드박스 Python 접근 실패는 데이터 생성 전 발생했고, 환경 변경 없이 실행 권한 적용으로 해결했다.','',
        '산출물: `results/laya_observation_controls_v1/`의 frozen_spec.md, protocol.json, source_snapshot, candidates.jsonl, data_audit.json, extended_data_audit.json, 각 실행 training/evaluation.json·trainable.pt, restoration_audit.json, verified_report.json.','',
        '한계: 학습32개와1024회라는 작은 부분 학습 실험이다. 같은 크기·토폴로지·ID 체계의 단일 장애만 다뤘다. 완전 중복 제외가 새로운 진단 규칙의 일반화를 보장하지 않는다. 전체 미세조정, 학습량 확대, 관측 표현의 추가 개선 효과와 모델 자체 능력 상한은 결정하지 못한다. 전체 도구 탐색은 재실행하지 않았다.']
    (PROJECT.parent/'docs/results/laya_observation_controls_v1.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    dump(OUT/'verified_report.json',dict(status='completed_verified',paired=paired,pattern_overlap_ignoring_latency=pattern_matches,
        runs=[dict(arm=r['arm'],seed=r['seed'],train=r['train']['metrics'],test=r['test']['metrics'],memorization_pass=r['memorization_pass'],new_case_pass=r['new_case_pass']) for r in runs],
        source_sha256={p.name:sha(p) for p in (PROJECT/'scripts').glob('*laya_observation*.py')}))
    for p in (PROJECT/'scripts').glob('*laya_observation*.py'):
        destination=OUT/'source_snapshot'/p.name
        if destination.exists():assert sha(destination)==sha(p)
        else:shutil.copyfile(p,destination)
    for r in runs:
        dump(OUT/f'{r["arm"]}_{r["seed"]}'/'status.json',
             {'status':'completed_verified','training':'not_applicable' if r['seed']=='base' else 'completed',
              'evaluation':'completed','restoration':'not_applicable' if r['seed']=='base' else 'completed_verified'})
    dump(OUT/'manifest.json',{'status':'completed_verified','completed_unix':time.time(),
        'phases':{'data':'completed_verified','training_runs':6,'evaluated_models':8,'restored_models':6,
                  'full_encoder_training':'not_run','laya_tool_rollouts':'not_run','paid_api_calls':0},
        'files':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}})
    print(json.dumps({'status':'completed_verified','paired':paired,'pattern_overlap':pattern_matches}))

if __name__=='__main__':main()
