"""Verify matched optimization experiments and report all predeclared runs."""
import json, statistics, itertools
import numpy as np
import torch
from safetensors import safe_open
from laya_learning_controls import BASE, PROJECT, dump, sha
from laya_optimization_controls import OUT, ARMS

def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    p=read(OUT/'protocol.json');runs=read(OUT/'summary.json')['runs']
    restored=read(OUT/'restoration_audit.json')
    assert restored['status']=='completed_verified' and len(restored['runs'])==12
    assert len(runs)==12 and {(r['arm'],r['seed']) for r in runs}==set(itertools.product(ARMS,[41,52,63]))
    assert sha(OUT/'source.py')==p['script_sha256'] and sha(BASE/'model.safetensors')==p['base_sha256']
    for split,digest in p['data_hashes'].items():assert sha(OUT/f'{split}.jsonl')==digest
    comparisons={'head_lr_1e3':{'head_lr'},'updates_1024':{'epochs'},'fixed_order':{'shuffle_options'}}
    for seed in p['seeds']:
        reference=read(OUT/f'reference_{seed}/config.json')
        historical=read(PROJECT/f'results/laya_symbol_controls_v1/last_block_{seed}/summary.json')
        current=next(r for r in runs if r['arm']=='reference' and r['seed']==seed)
        assert np.allclose(current['loss_history'],historical['loss_history'],rtol=1e-5,atol=1e-6)
        for arm,keys in comparisons.items():
            cfg=read(OUT/f'{arm}_{seed}/config.json')
            assert {k for k in cfg if cfg[k]!=reference[k]}==keys
    checks=[]
    for r in runs:
        t=OUT/f'{r["arm"]}_{r["seed"]}';curve=read(t/'curve.json');cfg=read(t/'config.json')
        assert r['status']=='completed' and curve[-1]==r['final']
        assert r['final']['epoch']==cfg['epochs'] and r['final']['updates']==16*cfg['epochs']
        assert len(r['loss_history'])==cfg['epochs']
        assert len(r['permutations'])==24 and {tuple(x['order']) for x in r['permutations']}==set(itertools.permutations(range(4)))
        assert abs(r['permutations'][0]['accuracy']-r['final']['validation']['accuracy'])<=.000051
        for row in curve:
            for split in ['train','validation']:
                v=row[split];z=np.asarray(v['logits']);y=np.asarray(v['targets'])
                assert np.isfinite(z).all() and abs(np.mean(z.argmax(1)==y.argmax(1))-v['accuracy'])<.0001
        weights=torch.load(t/'trainable.pt',weights_only=True,map_location='cpu')
        names=read(t/'trainable.json')['names'];assert set(names)==set(weights)
        assert all(n.startswith('encoder.layers.27.') for n in names if n.startswith('encoder.'))
        with safe_open(BASE/'model.safetensors',framework='pt') as original:
            changed={n:float((w-original.get_tensor(n).float()).abs().max()) for n,w in weights.items()}
        assert changed==r['parameter_changes'] and any(v>0 for n,v in changed.items() if n.startswith('encoder.'))
        assert sha(t/'trainable.pt')==r['checkpoint_sha256']
        checks.append(dict(arm=r['arm'],seed=r['seed'],changed_tensors=sum(v>0 for v in changed.values())))
        del weights
    lines=['# LAYA 학습률·학습 예산·선택지 순서 검증','',
        '## 검증할 가설','',
        '- H1: Head 학습률을 1e-4에서 1e-3으로 높이면 임의 신호 대응 학습이 개선되는가?',
        '- H2: 갱신 횟수를 256에서 1,024로 늘리면 개선되는가?',
        '- H3: 학습 중 선택지 순서를 고정하면 개선되며, 그 개선이 다른 선택지 순서에서도 유지되는가?', '',
        '각 변수를 기준 설정에서 하나씩 바꿔 seed 41/52/63에서 실행했다. 변수 간 상호작용은 검증하지 않았다. Encoder 마지막 블록과 Head만 학습했고 나머지 Encoder는 고정했다. Encoder LR 2.5e-5, soft-CE, FP32, microbatch 1, max_len/head_max_len 512, 동일한 기본 체크포인트와 학습 16개·검증 32개를 사용했다.', '',
        '학습·검증 모두 95% 이상을 사전 통과 기준으로 고정했다. 기준 설정은 16 epochs·256 updates·Head LR 1e-4·선택지 섞기다. 1,024 updates 조건은 64 epochs로, cosine learning-rate schedule도 길어지므로 갱신 횟수만의 순수한 효과는 아니다. Head 학습률 조건에서 Encoder 학습률은 바꾸지 않았다.', '',
        '입력은 이전과 동일한 사례 ID 및 amber/violet/silver/copper 신호다. 신호는 정답에서 만든 합성 변수이며 실제 장애 관측이 아니다. 반복 사용한 개발 검증셋이므로 이 점수는 새 네트워크 일반화 성능이나 최종 테스트 점수가 아니다.', '',
        '## 전체 결과','',
        '| 조건 | seed | 학습 정확도 | 검증 정확도 | 24순서 평균/최저 | 신호 제거 정확도 | 검증 CE | 학습+중간평가 초 |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    grouped={}
    for arm in ARMS:
        rr=sorted([r for r in runs if r['arm']==arm],key=lambda r:r['seed']);grouped[arm]=rr
        for r in rr:
            scores=[x['accuracy'] for x in r['permutations']];v=r['final']
            lines.append(f'| {arm} | {r["seed"]} | {v["train"]["accuracy"]:.2%} | {v["validation"]["accuracy"]:.2%} | {statistics.mean(scores):.2%}/{min(scores):.2%} | {r["signal_removed"]["accuracy"]:.2%} | {v["validation"]["loss"]:.4f} | {r["training_and_eval_seconds"]:.1f} |')
    effects={}
    lines+=['','## 가설별 판단','']
    for arm in comparisons:
        delta=[r['final']['validation']['accuracy']-ref['final']['validation']['accuracy'] for r,ref in zip(grouped[arm],grouped['reference'])]
        effects[arm]=dict(paired_validation_differences=delta,mean_difference=statistics.mean(delta),passes=sum(r['passed'] for r in grouped[arm]))
        lines.append(f'- {arm}: 같은 seed 기준 검증 정확도 변화 {[round(d*100,2) for d in delta]}%p, 평균 {statistics.mean(delta)*100:+.2f}%p. 사전 기준 통과 {effects[arm]["passes"]}/3.')
    budget=grouped['updates_1024']
    if all(r['final']['validation']['accuracy']>.25 for r in budget):
        lines.append('1,024회 조건은 세 seed 모두 기준 25%보다 높았다. 적어도 이 합성 과제에서는 기존의 짧은 학습 결과만으로 “새 대응을 배울 수 없다”고 결론 내릴 수 없다. 학습 예산/스케줄은 실패 원인 후보로 지지되며, 입력이나 모델 자체의 한계 하나로 설명할 수 없다.')
    for arm,rr in grouped.items():
        order_mean=statistics.mean(x['accuracy'] for r in rr for x in r['permutations'])
        removal=statistics.mean(r['signal_removed']['accuracy'] for r in rr)
        canonical=statistics.mean(r['final']['validation']['accuracy'] for r in rr)
        lines.append(f'- {arm}의 세 seed 평균: 정규 순서 {canonical:.2%}, 24순서 {order_mean:.2%}, 신호 제거 {removal:.2%}.')
        if arm=='fixed_order' and canonical-order_mean>.05:
            lines.append('선택지 고정의 정확도 개선은 순서를 바꾸면 상당 부분 사라진다. 정규 순서 점수만으로 신호 대응을 잘 배웠다고 평가하면 과대평가할 수 있다.')
        if arm=='updates_1024' and order_mean-removal>.2:
            lines.append('긴 학습 조건은 여러 선택지 순서에서도 신호 제거보다 높은 점수를 보였다. 합성 신호 사용을 지지하지만 최저 순서 점수를 보면 순서 민감도가 남아 있다. 신호 제거는 입력 분포 변화도 포함하므로 이것만으로 학습 메커니즘 전체를 확정하지 않는다.')
    if not any(r['passed'] for r in runs):
        lines.append('**사전 95% 기준을 통과한 실행은 없었다.** 다만 통과 여부와 학습 개선 여부는 다르다. 위 paired 차이, 선택지 순서 평가, 신호 제거를 함께 보아야 하며, 개선이 있어도 안정적으로 기준을 만족하는 설정을 확보한 것은 아니다.')
    else:
        lines.append('사전 기준을 통과한 조건이 있다. 다만 고정 선택지에서의 정확도만으로 신호 학습을 인정하면 안 된다. 24순서 평가와 신호 제거 결과를 함께 읽어야 하며, 실제 관측으로의 전이는 별도로 검증해야 한다.')
    lines+=['','## 결과를 읽는 기준과 재현','',
        '- 선택지 순서 24개는 같은 32개 사례의 반복 측정이다. 독립 표본 768개로 계산하지 않는다. 순서 평균·최저값은 위치 의존성 진단용이며 별도의 통과 기준으로 사후 변경하지 않았다.',
        '- 신호 제거는 사례 ID와 질문을 유지한 입력 개입이다. 원래 입력과의 정확도 차이는 신호 사용 여부를 점검하지만, 낯선 unknown 입력의 분포 변화도 포함한다.',
        '- 세 seed는 작은 반복 수다. 통계적 유의성이나 광범위한 하이퍼파라미터 최적성을 주장하지 않는다. 마지막 epoch 결과를 보고했고 중간 최댓값으로 선별하지 않았다.',
        '- 입력/소스/기본 체크포인트 해시, 조건 간 설정 차이, 모든 24순서, logits에서 재계산한 정확도, 저장 파라미터의 실제 변화와 해시를 검증했다. 기준 설정의 epoch별 학습 손실도 직전 실행과 일치했다.',
        '- trainable.pt에는 Head와 학습한 Encoder 마지막 블록이 함께 들어 있다. 12개 모두 기본 체크포인트에 다시 적용해 검증 logits 오차 1e-4 미만과 정확도 일치를 확인했다. 단독 laya.load용 내보내기는 아니다.',
        '- 시간에는 중간 평가가 포함된다. total_seconds에는 24순서 평가·신호 제거·저장도 포함된다. 비용은 API 호출 $0이며 전력 비용은 측정하지 않았다.',
        '- 실제 네트워크 장애 정확도, 오경보, 장애 위치, 탐색 단계 수, 지연·손실·처리량, 실장비 성능을 새로 측정한 실험은 아니다.', '',
        '다음 단계에서는 추가 탐색 전에 새로운 평가 사례와 통과 기준을 고정해야 한다. 개선된 학습 예산을 출발점으로 정답 노출 없이 동일한 실제 공개 관측을 표/문장으로 표현한 비교를 설계하되, 현재 합성 대조군의 순서 민감도와 기준 미달을 해결한 것으로 간주하지 않는다. 전체 Encoder 학습이 필요한지 여부도 이번 부분 학습 실험만으로 결정하지 않는다.', '',
        '`scripts/laya_optimization_controls.py` 실행 후 `scripts/verify_laya_optimization_restore.py`, `scripts/report_laya_optimization_controls.py` 순서로 검증한다. 계획·소스·입력·설정·학습 곡선·가중치·검증 기록은 `results/laya_optimization_controls_v1/`에 있다.', '',
        '[직전 입력/Encoder 대조 실험](laya_symbol_controls_v1.md)']
    dump(OUT/'verified_report.json',dict(status='completed_verified',effects=effects,checks=checks,api_cost_usd=0))
    formatted=[]
    for line in lines:
        if line and formatted and formatted[-1]:
            same_list=line.startswith('- ') and formatted[-1].startswith('- ')
            same_table=line.startswith('|') and formatted[-1].startswith('|')
            if not same_list and not same_table:formatted.append('')
        formatted.append(line)
    (PROJECT.parent/'docs/results/laya_optimization_controls_v1.md').write_text('\n'.join(formatted)+'\n',encoding='utf8')
    print(json.dumps(dict(status='completed_verified',effects=effects)),flush=True)

if __name__=='__main__':main()
