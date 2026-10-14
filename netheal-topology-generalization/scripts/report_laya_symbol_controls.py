"""Verify synthetic association experiments and document the resource interruption."""
import json
from collections import Counter
from laya_learning_controls import PROJECT, dump, sha
from laya_symbol_controls import OUT

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    p=read(OUT/'protocol.json'); result=read(OUT/'summary.json')
    assert result['status']=='completed' and len(result['runs'])==6
    assert sha(OUT/'source.py')==p['script_sha256']
    short=PROJECT/'results/laya_short_controls_v1'
    assert read(short/'interruption.json')['status']=='interrupted_resource_constraint'
    assert sha(short/'source.py')==read(short/'protocol.json')['script_sha256']
    for seed in [41,52,63]:
        s=read(short/f'frozen_{seed}/summary.json'); c=read(short/f'frozen_{seed}/curve.json')
        assert s['status']=='completed' and s['final']==c[-1]
        assert all(c[e][split]['accuracy']==1 for e in [0,-1] for split in ['train','validation'])
    assert not (short/'full_41/summary.json').exists()
    data={s:[json.loads(l) for l in (OUT/f'{s}.jsonl').read_text().splitlines()] for s in ['train','validation']}
    for s,n in [('train',4),('validation',8)]:
        assert set(Counter(r['expected']['fault'] for r in data[s]).values())=={n}
        assert all(r['expected']['fault'] not in r['state'] for r in data[s])
    assert not {r['episode_seed'] for r in data['train']} & {r['episode_seed'] for r in data['validation']}
    lines=['# LAYA 입력 축소·부분 Encoder 학습 검증','',
        '## 가설과 실행 범위','',
        'H1: 긴 관측 표를 없앤 짧은 정답 문장을 기본 모델이 읽을 수 있는가? H2: 정답을 직접 쓰지 않은 임의 신호와 장애 유형의 대응을 새로 학습할 수 있는가? H3: 동일 조건에서 Encoder 마지막 블록을 학습하면 Head만 학습할 때보다 개선되는가?', '',
        '먼저 짧은 정답 문장으로 Head 전용 학습을 세 seed에서 실행했다. 모두 학습 전·후 학습/검증 정확도가 100%였다. 이는 짧은 문장의 정답을 읽는 능력을 보여주지만 새로운 학습의 증거는 아니다. 전체 Encoder FP32 AdamW 실행은 첫 epoch 후 6GB GPU 자원 제약과 속도 저하로 중단했다. 완료된 전체 학습 성능을 주장하지 않는다. 원본 계획과 중단 기록은 `results/laya_short_controls_v1/`에 보존했다.', '',
        '후속 임의 신호는 device_failure→amber, link_failure→violet, misconfiguration→silver, performance_degradation→copper로 고정했다. 입력은 사례 ID와 신호만 포함하며 대응표를 질문에 제공하지 않았다. 신호 자체는 정답에서 만든 합성 신호이므로 실제 장애 증거가 아니다.', '',
        '학습 16개(유형별 4), 검증 32개(유형별 8), seed 41/52/63, 16 epochs=256 updates, soft-CE, FP32, microbatch 1, 선택지 순서 섞기, max_len/head_max_len 512를 고정했다. Head LR 1e-4, Encoder LR 2.5e-5. 조건은 Head 전용과 Head+Encoder 마지막 블록 학습이다. 나머지 Encoder 파라미터는 고정했다. 통과 기준은 학습·검증 모두 95% 이상이다.', '',
        '| 조건 | seed | 초기 학습/검증 | 최종 학습/검증 | 최종 검증 CE | 시간(초) | 최대 GPU 할당 GiB |',
        '|---|---:|---|---|---:|---:|---:|']
    checks=[]
    for seed in p['seeds']:
        a=read(OUT/f'frozen_{seed}/config.json'); b=read(OUT/f'last_block_{seed}/config.json')
        assert {k for k in a if a[k]!=b[k]}=={'freeze_encoder'}
    for r in result['runs']:
        target=OUT/f'{r["arm"]}_{r["seed"]}'; curve=read(target/'curve.json')
        assert [x['epoch'] for x in curve]==[0,4,8,16] and curve[-1]==r['final']
        delta=r['encoder_parameter_max_change']; trainable=read(target/'trainable.json')
        enc=[n for n in trainable['names'] if n.startswith('encoder.')]
        if r['arm']=='frozen': assert delta==0 and not enc
        else:
            assert delta>0 and enc and all(n.startswith('encoder.'+r['encoder_parameter'].split('.')[0]+'.'+r['encoder_parameter'].split('.')[1]+'.') for n in enc)
        a,b=curve[0],curve[-1]
        lines.append(f'| {r["arm"]} | {r["seed"]} | {a["train"]["accuracy"]:.1%}/{a["validation"]["accuracy"]:.1%} | {b["train"]["accuracy"]:.1%}/{b["validation"]["accuracy"]:.1%} | {b["validation"]["loss"]:.4f} | {r["seconds"]:.1f} | {r["peak_gpu_allocated_bytes"]/2**30:.2f} |')
        checks.append(dict(arm=r['arm'],seed=r['seed'],head_sha256=sha(target/'head.pt'),encoder_delta=delta,trainable_parameters=trainable['count']))
    passes={a:sum(r['passed'] for r in result['runs'] if r['arm']==a) for a in p['arms']}
    lines+=['','## 결과 해석','',f'사전 기준 통과: Head 전용 {passes["frozen"]}/3, 마지막 Encoder 블록 학습 {passes["last_block"]}/3.']
    if any(passes.values()):
        lines.append('임의 신호 대응에서 초기 대비 개선이 있는 실행은 현재 환경에서도 새 대응을 학습할 수 있다는 증거다. Head/부분 Encoder 조건의 차이는 위 세 seed 결과 범위에서 해석해야 하며, 전체 Encoder 학습의 효과로 바꿔 말할 수 없다.')
    else:
        lines.append('이번 학습 예산에서는 임의 신호 대응도 사전 기준을 통과하지 못했다. 최적화 조건과 표현을 더 분리해야 하며 모델 자체의 진단 능력 한계를 확정할 수 없다.')
    lines+=['','짧은 명시적 정답을 읽는 결과와 긴 관측 표의 대조군 실패가 다르므로 입력 구성의 영향을 추가 검증할 근거가 생겼다. 다만 이전 실험과 입력뿐 아니라 학습 예산도 달라 긴 입력이 유일한 원인이라고 단정할 수 없다.', '',
        '## 남은 검증과 제한','',
        '- 이 검증셋은 새 사례 ID에 같은 합성 규칙을 적용한 개발 대조군이다. 실제 네트워크 관측이나 미지 토폴로지에 대한 일반화 점수가 아니다.',
        '- 다음 단계는 정답을 사용하지 않고 동일한 공개 관측을 표/자연어로 변환한 뒤, 같은 예산과 분할에서 비교하는 것이다. 실제 관측 소수 사례 암기와 독립 검증을 통과한 뒤 도구 탐색을 재평가해야 한다.',
        '- 정상 상태 오경보, 복합 장애, 실장비, 지연·손실·처리량 등의 실제 네트워크 지표에 대한 새 성능 결론은 이 실험에서 내리지 않는다.',
        '- 실행 시간에는 중간 평가와 Head 저장이 포함된다. GPU 수치는 PyTorch 할당량이며 전용 VRAM·전력 비용이 아니다. API 호출 비용은 $0이다.',
        '- 부분 Encoder 실행의 파라미터 변화와 학습 대상 이름을 기록했다. head.pt만 저장하므로 부분 Encoder까지 포함한 배포 모델은 이 산출물로 복원할 수 없다. 진단 실험이며 필요하면 고정 소스로 재학습해야 한다.', '',
        '## 산출물','',
        '`scripts/laya_symbol_controls.py`, `scripts/report_laya_symbol_controls.py`, `results/laya_symbol_controls_v1/`에 소스·계획·입력·학습 곡선·logits·검증 기록을 보존했다. 손실, Brier, ECE, 확신도는 curve.json에 포함되지만 합성 소표본 결과를 운용 확신도로 사용하지 않는다.', '',
        '[이전 대조군](laya_learning_controls_20261010.md) · [공식 학습 안내](https://nandhakishorm.github.io/laya/finetune/)']
    dump(OUT/'verified_report.json',dict(status='completed_verified',passes=passes,checks=checks,input_hashes={s:sha(OUT/f'{s}.jsonl') for s in data}))
    (PROJECT.parent/'docs/results/laya_symbol_controls_v1.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status='completed_verified',passes=passes)))

if __name__=='__main__': main()
