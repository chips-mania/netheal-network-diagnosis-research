"""Verify and summarize the short-input frozen/full encoder diagnostic."""
import json
from pathlib import Path
from laya_learning_controls import sha, dump
from laya_short_controls import OUT, PROJECT

def read(p): return json.loads(p.read_text(encoding='utf8'))

def main():
    protocol=read(OUT/'protocol.json'); summary=read(OUT/'summary.json')
    assert summary['status']=='completed' and len(summary['runs'])==6
    assert sha(OUT/'source.py')==protocol['script_sha256']
    rows={s:[json.loads(l) for l in (OUT/f'{s}.jsonl').read_text().splitlines()] for s in ['train','validation']}
    assert not {r['episode_seed'] for r in rows['train']} & {r['episode_seed'] for r in rows['validation']}
    assert len(rows['train'])==16 and len(rows['validation'])==32
    for seed in protocol['seeds']:
        a=read(OUT/f'frozen_{seed}/config.json'); b=read(OUT/f'full_{seed}/config.json')
        assert {k for k in a if a[k]!=b[k]}=={'freeze_encoder'}
    lines=['# LAYA 짧은 입력과 Encoder 학습 대조 실험','',
        '## 검증 질문과 조건','',
        '짧은 정답 노출 문장에서 Encoder를 고정할 때와 전체 학습할 때 학습 가능성이 달라지는가? 기존 관측 표는 제거하고 사례 ID와 정답 유형만 남겼다. 이 점수는 네트워크 진단 정확도가 아니다.', '',
        '학습 16개·검증 32개, 장애 4종 균형, seed 41/52/63, soft-CE, FP32, 16 epochs(256 updates), microbatch 1, 선택지 순서 섞기를 사용했다. Encoder LR 2.5e-5, Head LR 1e-4이다. 두 조건의 TrainConfig 차이는 freeze_encoder 하나뿐이다. 평가 후 각 조건의 train/eval 모드를 복원했다.', '',
        '통과 기준은 학습·검증 모두 95% 이상이다. 이전 긴 입력 실험은 512 updates였으므로, 이전 결과와의 차이를 입력 길이 하나의 인과 효과로 해석하면 안 된다. 이번 두 조건 사이에서만 통제 비교가 가능하다.', '',
        '| 조건 | seed | 초기 학습/검증 | 최종 학습/검증 | Encoder 검사 파라미터 최대 변화 | GPU 최대 할당 GiB | 시간(초) |',
        '|---|---:|---|---|---:|---:|---:|']
    checks=[]
    for run in summary['runs']:
        target=OUT/f'{run["arm"]}_{run["seed"]}'
        curve=read(target/'curve.json'); assert [r['epoch'] for r in curve]==[0,4,8,16]
        assert curve[-1]==run['final']
        delta=run['encoder_parameter_max_change']
        assert delta==0 if run['arm']=='frozen' else delta>0
        a,b=curve[0],curve[-1]
        lines.append(f'| {run["arm"]} | {run["seed"]} | {a["train"]["accuracy"]:.1%}/{a["validation"]["accuracy"]:.1%} | {b["train"]["accuracy"]:.1%}/{b["validation"]["accuracy"]:.1%} | {delta:.6g} | {run["peak_gpu_allocated_bytes"]/2**30:.2f} | {run["seconds"]:.1f} |')
        checks.append(dict(arm=run['arm'],seed=run['seed'],head_sha256=sha(target/'head.pt')))
    passes={arm:sum(r['passed'] for r in summary['runs'] if r['arm']==arm) for arm in protocol['arms']}
    lines+=['','## 해석과 한계','',f'사전 기준 통과: Encoder 고정 {passes["frozen"]}/3, 전체 학습 {passes["full"]}/3.']
    if all(read(OUT/f'{r["arm"]}_{r["seed"]}/curve.json')[0]['validation']['accuracy']==1 for r in summary['runs']):
        lines.append('**모든 실행이 학습 전부터 검증 100%였다. 따라서 최종 100%는 새로운 규칙을 학습했다는 증거가 아니다.** 손실 감소는 정답에 대한 점수 차이가 커졌음을 보여줄 수 있지만, 정확도 향상이나 새로운 과제 학습을 검증하려면 임의 신호 대응 대조군이 필요하다.')
    if passes['full']>passes['frozen']:
        lines.append('이번 짧은 합성 과제에서는 Encoder까지 학습하는 조건이 더 많은 seed에서 통과했다. Head만 학습한 기존 실패를 모델 자체의 능력 한계로 일반화할 수 없다. 실제 관측에도 이 개선이 전이되는지는 별도 검증해야 한다.')
    elif not any(passes.values()):
        lines.append('짧은 입력과 전체 학습에서도 이번 예산으로는 대조군을 통과하지 못했다. 따라서 입력 길이나 Encoder 고정 중 하나만 원인이라고 단정할 수 없다. 최적화·표현·학습 구현에 대한 추가 분리가 필요하다.')
    else:
        lines.append('짧은 정답 신호에서 통과 기준을 만족하는 조건이 확인되었다. 초기 점수와 비교해야 하며, 이 결과만으로 새로운 학습이나 실제 네트워크 관측에서의 학습을 보장하지는 않는다.')
    lines+=['','검증 문장에도 정답을 의도적으로 노출했다. 새 사례 ID를 사용하지만 같은 정답 문장 틀이므로 일반화 성능 검증이 아니다. 손실·Brier·ECE·확신도와 logits는 각 curve.json에 기록했다. 소표본·정답 노출 조건의 확신도를 실제 운용에 적용하면 안 된다.', '',
        '시간에는 중간 평가와 Head 저장이 포함된다. GPU 수치는 PyTorch 최대 할당량이며 실제 전용 VRAM 사용량이나 전력 비용이 아니다. 전체 학습 실행은 Encoder 검사 파라미터의 변화로 검증했으며, 저장한 head.pt만으로 전체 학습 모델을 복원할 수는 없다. 이 실행은 배포용 모델 생산이 아닌 학습 경로 진단이다.', '',
        'API 비용 $0. 실제 관측의 자연어 변환, 장애 위치 분류, 도구 탐색 재평가, 정상 상태 오경보·복합 장애·실장비 평가는 이 실행에 포함되지 않는다.', '',
        '## 재현','',
        '`scripts/laya_short_controls.py`를 실행한 뒤 `scripts/report_laya_short_controls.py`로 검증한다. 입력, 소스 스냅샷, 계획, 설정, 학습 곡선은 `results/laya_short_controls_v1/`에 보존한다.', '',
        '[공식 학습 안내](https://nandhakishorm.github.io/laya/finetune/) · [이전 대조 실험](laya_learning_controls_20261010.md)']
    dump(OUT/'verified_report.json',dict(status='completed_verified',passes=passes,checks=checks,data_hashes={s:sha(OUT/f'{s}.jsonl') for s in rows}))
    (PROJECT.parent/'docs/results/laya_short_controls_v1.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status='completed_verified',passes=passes)),flush=True)

if __name__=='__main__':main()
