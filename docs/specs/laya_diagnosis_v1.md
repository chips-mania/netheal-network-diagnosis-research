# LAYA 진단 모델 개발 v1

2026-10-09. GPT/Jev 추가 호출 없이 진행한다. 기존 API pilot 결과와 입력·질문 구조가 달라 직접 우열 비교하지 않는다.

## 가설

- L1: 공식 공개 체크포인트가 의미를 명시한 공개 관측과 분리된 질문에서 유효한 행동을 선택할 수 있는가?
- L2: 같은 LAYA 체크포인트를 전문가 궤적으로 미세조정하면, 독립 에피소드에서 행동 모방과 실제 진단 성공률이 개선되는가?
- L3: 개선이 이름 변경·학습하지 않은 토폴로지에도 유지되는가?

기본 모델 점검과 학습 smoke는 L2/L3의 검증 완료가 아니다.
교사 행동 일치율은 진단 정확도가 아니며, 전문가 행동은 여러 유효한 행동 중 하나일 수 있다.

## 고정된 개발 설정

- laya==0.4.1, convaiinnovations/laya revision 7b928d828b7b0e022f929d9bd2e44165aa270148.
- 별도 .runtime/laya-venv, 모델 캐시는 .runtime/huggingface. CPU 4 threads.
- 공개 그래프의 모든 기존 특징을 이름과 함께 전달한다. 정답·교사추천은 입력하지 않는다.
- 질문은 kind / 조사 node / ping edge / 진단 fault / 진단 node / 진단 edge로 분리한다.
- 학습에서는 실제 전문가 행동에 해당하는 질문에만 정답을 붙인다. 학습 평가에서 정답 분기로 필요한 질문을 고르는 것은 teacher forcing임을 명시한다.
- 입력 max_len 4096, head_max_len 1024. 공식 tokenizer/build_sequence로 state 잘림, 후보 탈락·합침을 검사하고 발생 시 중단한다.
- 기본 모델 32개 교사 관측 평가: validation.jsonl에서 균일 간격으로 선택, 결과를 보기 전에 고정.
- 자체 학습 여부를 나누어 보고하고 API 호출 비용 0과 연산 비용 미측정을 구분한다.

## 데이터

기존 v3 mixed replica 41 전문가 2,112 에피소드(Star/Linear, 크기4/5/6, 네 장애 유형)를 사용한다.
각 topology×size×fault 88개 에피소드를 고정 seed712로 섞고 64/12/12개로 나눈다.

| 분할 | 에피소드 | 단계 |
|---|---:|---:|
| train | 1536 | 8143 |
| calibration | 288 | 1556 |
| validation | 288 | 1535 |

같은 에피소드의 관측은 반드시 같은 분할에 둔다. 서로 다른 장애의 초기 미관측 상태가 같은 것은 정상이며 독립적인 텍스트 입력 분할이라는 뜻은 아니다.
API pilot 64개와 본 평가 예약 seed는 학습에 쓰지 않는다. 분할 목록과 원본/출력 해시를 보존한다.

## 학습 및 평가

CPU에서는 먼저 8개 질문에 대한 frozen-encoder 학습 smoke를 수행해 gradient와 시간을 확인한다.
이는 충분한 미세조정 결과가 아니다. 본 학습 후보는 전체 encoder 학습, 4epochs, microbatch1×accum16, 세 seed41/52/63이다.
학습 전 GPU 메모리·실행 시간을 점검하고 예산을 확정한다. 별도 calibration 분할로 보정하며 validation을 보정에 재사용하지 않는다.
초기 checkpoint의 temperature 경고가 있으므로 현재 confidence를 보정된 성공 확률로 간주하지 않는다.

최종 모델 선택에는 validation을 사용하고, 최종 성능은 별도 새 사례에서 실제 도구를 선택하는 rollout으로 측정한다.
후보 순서/ID 변경, 미완료·반복 행동, Exact·장애별F1, CPU/GPU 추론 지연·메모리·학습시간·서버비용을 보고한다.
GPU/CPU 로컬 지연과 GPT/Jev API 왕복 지연을 같은 종류의 추론 속도라고 표현하지 않는다.

## 실행

프로젝트의 netheal-topology-generalization 폴더에서, PYTHONPATH에 netheal-ai-agent-benchmark를 설정한다.

```powershell
../.runtime/laya-venv/Scripts/python.exe -m src.prepare_laya_data --out results/laya_data_v1
../.runtime/laya-venv/Scripts/python.exe -m src.check_laya_baseline --out results/laya_baseline_new --limit 32
../.runtime/laya-venv/Scripts/python.exe -m src.train_laya_diagnosis --smoke --out results/laya_train_smoke_new
# 준비된 전체 학습 명령; CUDA 장비에서만 실행
python -m src.train_laya_diagnosis --device cuda --base PATH_TO_PINNED_CHECKPOINT --seed 41 --out results/laya_ft_41
```

전체 학습 wrapper는 학습 checkpoint까지만 저장한다. 별도 보정과 실제 rollout 검증은 이어서 구현/실행해야 한다.
[공식 학습 안내](https://nandhakishorm.github.io/laya/finetune/)


## 2026-10-09 실행 중 보완: GPU와 compact v2

- 실제 장비는 NVIDIA GTX 1660 SUPER 6GB, RAM 약 16GB이다. GPU가 없다는 초기 판단은 CPU 전용 PyTorch를 사용한 데서 나온 오류였다. LAYA 전용 환경에 torch 2.12.1+cu126을 설치했고 GPU 연산을 검증했다. 기존 연구 환경은 변경하지 않았다.
- 전체 encoder 학습 점검 8개 질문은 81.41초에 완료됐지만 PyTorch 최대 할당량은 8.35GB였다. 이는 물리 VRAM 6GB를 초과하며 Windows 공유 메모리 사용 가능성이 있어, 순수 VRAM 안에서 원활하게 학습한다고 해석하지 않는다.
- 반복된 특징 이름을 한 번만 쓰는 compact-table-v2를 추가했다. 모든 수치 특징을 기존 5자리 유효숫자 정밀도로 유지한다. 초기 평가 64개 사례의 384개 질문에서 잘림 0건, 최대 2010 tokens를 확인했다. 매 실제 추론에서도 잘림을 다시 검사한다.
- 긴 입력 v1의 GPU rollout은 완료되지 않은 입력 형식 점검으로 보존한다. 중단 상태를 manifest에 기록했고 정확도 결과로 사용하지 않는다.
- 소규모 학습은 train 분할에서 24개 topology×size×fault 조합별 2개씩 총 48개 에피소드, 516개 질문, 2 epochs, seed 41로 고정했다. encoder를 고정하고 head만 학습하며 전체 미세조정과 구분한다. validation/calibration/API pilot 사례는 학습에 넣지 않는다.
- 학습 전후 compact v2를 같은 64개 API pilot 사례, 40회 행동 제한으로 평가한다. 분기 질문은 모델이 선택한 행동과 장애 유형으로 정한다. 정답 분기를 사용하지 않는다.
- 중단 후 --resume으로 완료한 사례를 건너뛸 수 있다. checkpoint/입력 형식/소스 해시가 달라지면 재개를 거부한다.
- confidence 보정은 아직 수행하지 않았다. 이번 단계는 argmax 행동과 실제 진단 결과를 평가하며 confidence calibration 가설을 검증하지 않는다.
- 정확도뿐 아니라 장애 유형 macro-F1, 위치 정확도, 오진/미완료/입력 용량 실패, 도구 횟수·반복·가상 비용, 평균/p95 실행 시간, 학습 시간, GPU 할당량을 기록한다. API 비용 0은 전력·장비 비용 0을 뜻하지 않는다.


### 실제 재현 명령 (compact v2, FP32)

```powershell
# netheal-topology-generalization 폴더에서 실행
$env:PYTHONPATH = (Resolve-Path ../netheal-ai-agent-benchmark).Path
../.runtime/laya-venv/Scripts/python.exe -m src.prepare_laya_data --compact --out results/laya_data_compact_v2
../.runtime/laya-venv/Scripts/python.exe -m src.audit_laya_capacity
../.runtime/laya-venv/Scripts/python.exe -m src.train_laya_diagnosis --device cuda --freeze-encoder --data-dir results/laya_data_compact_v2 --train-episodes 48 --epochs 2 --out results/laya_head_compact_48ep_v2
../.runtime/laya-venv/Scripts/python.exe -m src.run_laya_comparison --compact --out results/laya_baseline_compact_fp32_v2
../.runtime/laya-venv/Scripts/python.exe -m src.run_laya_comparison --compact --checkpoint results/laya_head_compact_48ep_v2/checkpoint --out results/laya_head_compact_fp32_v2
../.runtime/laya-venv/Scripts/python.exe scripts/report_laya_pilot.py
../.runtime/laya-venv/Scripts/python.exe scripts/verify_laya_checkpoint.py
```

기존 결과 디렉터리를 덮어쓰지 않는다. 새 실험에는 새 경로를 쓰고, 동일 평가 중단 후에는 해당 평가 명령에 `--resume`을 추가한다. baseline과 학습 모델을 동시에 GPU에서 실행하지 않는다. 지연 측정 시 다른 GPU 작업을 겹치지 않는다.

초기 GPU 추론의 FP16 점검은 중단 기록으로 분리했다. 최종 paired pilot은 FP32로 일치시키며, 기본 모델의 첫 동일 사례에서 관측한 76초→13초 차이는 단일 사례의 환경 설정 점검이지 일반적인 속도 향상 추정치가 아니다.


### 추가 원인 분리 평가

실제 탐색 결과를 보고 도구 선택 실패와 최종 판단 실패를 구분하기 위해 별도 보조 평가를 추가한다. validation의 마지막 전문가 진단 상태 288개(학습과 에피소드 비중복)를 사용하며, LAYA가 장애 유형을 예측한 뒤 그 예측에 해당하는 위치 질문을 받는다. 정답 유형은 평가할 때만 사용한다. 기본/학습 모델을 같은 상태에서 비교한다. 이 자료는 전문가가 성공한 Star/Linear의 크기 4–6 사례라 실제 pilot 64개와 모집단이 다르다. 두 정확도의 차이만으로 원인을 인과적으로 확정하지 않는다.

원래 checkpoint 기본 입력 예산은 512/192이며 이번 4096/1024 확장은 잘림 방지를 위한 설정이다. 긴 입력과 숫자 표에 대한 모델 적응은 별도 성능 문제로 남는다.


## 최종 실행 상태 (2026-10-09)

계획했던 이번 pilot의 남은 평가는 완료했다. 실제 탐색은 기본/Head 학습 모델 각 64건(총 128건), 증거 수집 후 진단은 각 288건(총 576건)이다. 결과·한계는 [최종 보고서](../results/laya_pilot_v2.md)를 따른다. GPU fallback과 입력 잘림은 없었다. 학습 체크포인트는 encoder 170개 tensor가 동일하고 head 36개 중 31개 tensor가 바뀐 것을 확인했다. 전체 encoder 미세조정·confidence 보정·여러 학습 seed 반복은 이번 완료 범위가 아니다.
