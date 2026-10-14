"""Report completed learning controls without confusing leaked-label scores with task accuracy."""
import json,hashlib
from pathlib import Path
from collections import Counter
import torch
from safetensors import safe_open

PROJECT=Path(__file__).resolve().parents[1];ROOT=PROJECT.parent
BASE=ROOT/'.runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148/model.safetensors'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    report=[];checks=[]
    for folder,expected_n in [('laya_learning_controls_20261010',6),('laya_learning_controls_ce_20261010',3)]:
        out=PROJECT/'results'/folder;p=read(out/'protocol.json');summ=read(out/'summary.json')
        assert summ['status']=='completed' and len(summ['runs'])==expected_n
        assert sha(out/'source.py')==p['script_sha256']
        assert not set(p['training_seeds'])&set(p['validation_seeds'])
        for split,digest in p['data_hashes'].items():
            assert sha(PROJECT/'results/laya_data_compact_v2'/f'{split}.jsonl')==digest
        for arm in p['arms']:
            for split,perclass in [('train',4),('validation',8)]:
                rows=[json.loads(l) for l in (out/f'{arm}_{split}.jsonl').read_text().splitlines()]
                assert set(Counter(r['expected']['fault'] for r in rows).values())=={perclass}
        for run in summ['runs']:
            assert run['final']['epoch']==32 and run['final']['optimizer_updates']==512
            target=out/f'{run["arm"]}_{run["seed"]}'
            head=torch.load(target/'head.pt',weights_only=True,map_location='cpu')
            with safe_open(BASE,framework='pt') as base:
                changed=[k for k,v in head.items() if not torch.equal(v,base.get_tensor(k).float())]
            assert changed
            checks.append(dict(run=str(target.relative_to(PROJECT)),changed_tensors=len(changed),head_sha256=sha(target/'head.pt')))
            report.append(dict(loss=p['loss'],arm=run['arm'],seed=run['seed'],
                train_accuracy=run['final']['train']['accuracy'],validation_accuracy=run['final']['validation']['accuracy'],
                passed=run['passed'],seconds=run['seconds']))
    out=PROJECT/'results/laya_learning_controls_20261010'
    ce=PROJECT/'results/laya_learning_controls_ce_20261010'
    for split in ('train','validation'):
        assert sha(out/f'label_tag_{split}.jsonl')==sha(ce/f'label_tag_{split}.jsonl')
    for seed in (41,52,63):
        a=read(out/f'label_tag_{seed}'/'config.json');b=read(ce/f'label_tag_{seed}'/'config.json')
        assert {k for k in a if a[k]!=b[k]}=={'loss'}
    audit=read(out/'gradient_audit.json')
    assert audit['nonzero_gradient_tensors']>0 and audit['parameter_max_change']>0 and audit['logit_max_change']>0
    data=read(out/'data_audit.json')
    assert all(data[s]['contradictory_state_labels']==0 for s in ('train','validation'))
    result=dict(status='completed_verified',runs=report,weight_checks=checks,gradient_audit=audit,data_audit=data,
        loss_comparison_identical_inputs=True,loss_comparison_only_config_difference='loss',
        api_cost_usd=0,base_sha256=sha(BASE),warning='These are learning controls, not end-to-end diagnosis benchmarks.')
    (out/'verified_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# LAYA 학습 가능 여부 검증','', '2026-10-10. 기존 pilot을 덮어쓰지 않고 수행한 진단용 실험이다. **정답을 노출한 대조군 점수는 네트워크 진단 성능이 아니다.**','',
      '## 가설과 고정 조건','',
      '- H1: 기존 Head 학습 설정이 입력에 명시된 쉬운 정답 신호라도 학습할 수 있는가?',
      '- H2: 실제 네트워크 관측 16개를 학습 데이터 자체에서 맞힐 수 있는가?',
      '- H3(후속): 정답 대조군에서 손실 함수만 RLCD → soft-CE로 바꾸면 달라지는가?',
      '- 기존 train 분할의 최종 전문가 진단 상태에서 장애별 4개씩 16개, validation에서 장애별 8개씩 32개를 고정 선택했다. 에피소드 비중복이며 검증 분할은 개발용이다.',
      '- 유형 분류 한 질문만 학습했다. 기본 checkpoint, Encoder 고정, FP32, Head LR 1e-4, microbatch 1, accumulation 1, 32 epochs=512 optimizer updates, 선택지 순서 섞기, seed 41/52/63을 사용했다.',
      '- 평가 시점은 0/4/8/16/32 epochs로 미리 고정했다. 평가 후 Head는 train, Encoder는 eval 모드로 복구했다.',
      '- 정답 대조군은 train/validation 모두 관측 앞에 정답 유형을 명시했다. 실제 표 조건에는 이 문장을 넣지 않았다.',
      '- 사전 통과 기준: 정답 대조군은 학습·검증 모두 95% 이상, 실제 관측 암기는 학습 정확도 95% 이상. 16개 학습 표본에서는 사실상 16/16을 요구한다.',
      '- 원래 pilot의 다중 질문 학습과는 과제 및 갱신 횟수가 다르므로 직접적인 성능 재현 실험이 아니다. 이번 조건끼리만 비교한다.','',
      '## 결과','', '| 손실 | 입력 | seed | 학습 정확도 | 검증 정확도 | 통과 |','|---|---|---:|---:|---:|---|']
    for r in report:
        lines.append(f'| {r["loss"]} | {r["arm"]} | {r["seed"]} | {r["train_accuracy"]*100:.2f}% | {r["validation_accuracy"]*100:.2f}% | {"예" if r["passed"] else "아니오"} |')
    lines+=['','## 실행 및 데이터 검증','',
      '- 클래스 균형·에피소드 분리·태그 격리 테스트 통과. 선택된 학습 데이터의 동일 관측에 상충하는 정답은 없었다.',
      f'- 서로 다른 에피소드지만 학습·검증의 관측 문자열이 같은 검증 사례가 {data["exact_observation_overlap_between_splits"]}개 있었다. 검증셋은 독립 에피소드 분할이며 완전히 새로운 관측 문자열 분할은 아니다. 이 소규모 검증 정확도를 일반화 성능으로 주장하지 않는다.',
      '- 공식 tokenizer로 입력 잘림과 선택지 손실을 검사한 뒤 실행했다. 실행 소스, 계획, 선택 사례, epoch별 정확도·혼동행렬, Head 가중치를 저장했다.',
      f'- 단일 soft-CE 갱신 점검에서 {audit["nonzero_gradient_tensors"]}개 텐서의 gradient가 0이 아니었고, Encoder gradient는 없었다. 가중치 최대 변화 {audit["parameter_max_change"]:.6g}, 출력 점수 최대 변화 {audit["logit_max_change"]:.6g}.',
      f'- 평가 fast-path 사용 여부에 따른 점수 최대 차이는 {audit["eval_fastpath_max_difference"]:.6g}였다. 이 한 사례의 점검만으로 모든 라이브러리 오류를 배제하지는 않는다.',
      '- 모든 완료된 실행에서 저장된 Head 가중치가 원본과 다른 것을 검증했다. 단순히 학습이 실행되지 않은 상황은 아니다.',
      '- RLCD 대조군 두 seed 완료 후 gradient 점검을 위해 실행을 중단했다가 남은 실행을 재개했다. 완료된 실행은 재선별하지 않았다.',
      '- API 호출 비용은 $0이며 전력·장비 비용은 측정하지 않았다. 실행 시간에는 중간 평가와 Head 저장이 포함돼 이전 학습 시간과 직접 비교하지 않는다.','',
      '## 해석','']
    positives=[r for r in report if r['arm']=='label_tag'];real=[r for r in report if r['arm']=='real_table']
    lines.append(f'정답 대조군 통과 {sum(r["passed"] for r in positives)}/{len(positives)}회, 실제 관측 암기 통과 {sum(r["passed"] for r in real)}/{len(real)}회.')
    if not any(r['passed'] for r in positives):
        lines.append('**현재 설정은 쉬운 정답 신호를 학습하는 기본 관문도 통과하지 못했다. 기존 0/64 결과를 LAYA의 네트워크 진단 능력 한계로 해석할 근거가 부족하다.** Encoder 고정의 영향, 입력 표현 및 최적화 조건을 더 분리해야 한다. 더 큰 실제 탐색 평가를 반복하기 전에 학습 대조군부터 통과시켜야 한다.')
    else:
        lines.append('정답 대조군의 통과는 해당 설정으로 쉬운 신호를 학습할 수 있음을 뜻한다. 실제 관측 학습이나 미지의 네트워크로의 일반화가 검증됐다는 뜻은 아니다. 과제·표현·Encoder 고정의 영향을 별도로 비교해야 한다.')
    lines+=['','이번 단계에서는 전체 Encoder 미세조정, 자연어 증거와 표의 비교, 위치 분류, 실제 도구 탐색 재평가를 완료했다고 주장하지 않는다. 정상 상태·복합 장애·실장비 성능 역시 검증하지 않았다.','',
      '## 재현과 산출물','',
      '- `scripts/laya_learning_controls.py`: RLCD 대조군 및 실제 관측 암기, 완료된 run은 재개 시 건너뜀.',
      '- `scripts/laya_gradient_audit.py`: 단일 갱신 및 평가 연산 경로 점검.',
      '- `scripts/laya_learning_controls_ce.py`: 동일 정답 대조군의 soft-CE 후속 실험.',
      '- `scripts/report_laya_learning_controls.py`: 완료 여부·데이터·소스·가중치 검증 후 보고서 생성.',
      '- `results/laya_learning_controls_20261010/`, `results/laya_learning_controls_ce_20261010/`: 원시 계획, 입력, 학습 곡선, 가중치 및 결과.',
      '- [공식 학습 및 양성 대조 안내](https://nandhakishorm.github.io/laya/finetune/).']
    (ROOT/'docs/results/laya_learning_controls_20261010.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],runs=report),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
