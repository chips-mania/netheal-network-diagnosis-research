# NetHeal Network Diagnosis Research

네트워크 장애 진단 에이전트가 **도구로 증거를 수집하고 장애 유형·위치를 판단하는 절차를 학습할 수 있는지**, 그리고 학습한 절차가 **다른 네트워크 크기와 토폴로지로 전이되는지** 검증하는 연구 저장소입니다.

## 목적

NetHeal 시뮬레이터 위에서 강화학습(PPO), 전문가 궤적 모방학습(Behavior Cloning, BC), GNN+pointer 정책을 비교합니다. 단순 성공률뿐 아니라 도구 사용, 성급한 진단, 타임아웃, 증거가 충분한데도 진단을 제출하지 않는 현상까지 분석합니다.

현재 중심 연구는 **네트워크 장애 진단 정책의 실패 모드와 크기 일반화**입니다. Cisco telemetry 기반 정상 패턴 전이 및 분포 변화 대응은 별도 연구 계획으로 보존되어 있으며, 이 저장소의 NetHeal 실험 성과와 구분합니다.

## 내용

- **환경 감사:** 관측·보상·액션 마스크를 점검하고 PPO가 `scan → diagnose`에 머무르는 원인을 분석합니다.
- **학습 프로토콜 비교:** PBRS, 타임아웃 패널티, 탐색, 진단 게이트, 관측 래퍼, Rule 기반 BC를 비교합니다.
- **일반화 실험:** Star 크기를 바꾸고, MLP와 GNN+pointer를 비교하며, Mesh 전이를 평가합니다.
- **병목 분해:** 진단 제출(commit)과 증거에 맞는 대상 노드 선택(pointer binding)을 분리해 분석합니다.

## 결과

아래 수치는 **이미 저장된 실험 산출물**의 요약이며, 이번 디렉터리 정리 과정에서 재학습한 결과가 아닙니다. Exact는 장애 **유형과 위치를 모두 맞힌 비율**입니다. 주로 단일 학습 seed와 조건별 100개 평가 에피소드로 수행한 PoC이므로 다중 seed 검증이나 실환경 성능을 뜻하지 않습니다.

- **Rule과 BC:** Star 6에서 GT를 행동 선택에 사용하지 않는 Rule은 Exact 100%. 현재 저장된 BC-rule 결과도 **100/100, 평균 5.86 step**입니다. 과거 취합 문서의 99%와 현재 결과 파일이 달라 여기서는 개별 결과 파일을 우선합니다. [Rule 보고서](netheal-topology-generalization/reports/rule_ceiling_poc.md) · [BC 보고서](netheal-topology-generalization/reports/bc_rule_poc.md) · [BC 원자료](netheal-topology-generalization/results/bc_rule_poc.json)
- **GNN의 부분적 크기 일반화:** Star {4,5,6} 전문가 데모로 학습한 P1 GNN+pointer는 Star 6/8/10에서 **100% / 90% / 42%**. 같은 비교의 B1 MLP는 **89% / 76% / 57%**입니다. Star 8에서는 +14%p지만 Star 10에서는 더 낮아 완전한 크기 일반화로 해석할 수 없습니다. [비교 보고서](netheal-topology-generalization/reports/graph_expert_poc.md)
- **진단 제출 병목 개선:** Generic commit supervision(진단 샘플 ×3, 진단 가중치 ×4)은 Star 10 Exact를 **42% → 66%**, Link 성공률을 **0% → 92%**로 개선했습니다. 다만 Star 8 Exact는 **90% → 86%**이고, 해당 평가의 Star 10 Misconfiguration은 **0%**로 남습니다. [개입·절제 보고서](netheal-topology-generalization/reports/p1_commit_ablation.md)
- **미해결 과제:** Star-only 학습 P1의 Mesh 6 Exact는 **9%**입니다. 토폴로지 계열 전이, 큰 Star에서의 Misconfiguration 대상 선택, 다중 seed 재현은 아직 해결되지 않았습니다. [전체 결과 취합](docs/results/GNN_expert_guided_results_report.md)

연구의 현재 성과는 **관측된 PPO 실패 모드를 정리하고, 전문가 절차 모방의 상한과 일반화 병목을 분리한 것**입니다. 보상·마스크를 바꾼 실험은 공식 원본 조건과 다른 실험 변형으로 해석해야 합니다.

## 디렉터리 구조

```text
.
├── README.md                         # 목적·내용·결과와 실행 안내
├── docs/
│   ├── README.md                     # 전체 문서 인덱스
│   ├── plans/                        # 연구 계획
│   ├── specs/                        # 현재 실험 설계서
│   ├── results/                      # 전체 결과 요약과 실험 이력
│   ├── archive/specs/                # 수정 전 설계서
│   └── external-repositories.md      # 외부 저장소 출처·고정 버전
├── netheal-topology-generalization/  # 자체 실험 코드
│   ├── config/                       # 학습·평가 프로토콜
│   ├── src/                          # 정책·래퍼·실험 실행 모듈
│   ├── reports/                      # 실험별 분석 보고서
│   ├── results/                      # JSON·CSV·에피소드 기록
│   └── models/checkpoints/           # 저장된 모델 체크포인트
├── netheal-ai-agent-benchmark/       # NetHeal 원본 환경 (submodule)
├── OutlierDenStream-BigDama18/       # 이상탐지 참고 구현 (submodule)
└── telemetry/                       # Cisco 원본 데이터 (submodule)
```

외부 저장소는 기존 경로와 이력을 유지하고 커밋을 고정합니다. 코드의 상대경로를 보존하기 위해 실행 코드·결과·체크포인트의 위치는 유지했습니다. 캐시, 가상환경, 비밀 설정, 임시 체크포인트는 Git 추적에서 제외합니다.

## 시작하기

### 1. 복제 및 환경 준비

NetHeal 실험만 실행할 때는 대용량 telemetry 데이터 없이 환경 서브모듈만 받으면 됩니다. 아래는 저장소에 접근 권한이 있는 계정의 PowerShell 예시입니다.

```powershell
git clone https://github.com/chips-mania/netheal-network-diagnosis-research.git
cd netheal-network-diagnosis-research
git submodule update --init netheal-ai-agent-benchmark

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r netheal-ai-agent-benchmark/requirements.txt
python -m pip install -r netheal-topology-generalization/requirements.txt

$env:PYTHONPATH = (Resolve-Path .\netheal-ai-agent-benchmark).Path
cd netheal-topology-generalization
```

기존 작업 환경은 Conda `netheal_base`입니다. 정리 시 확인한 환경은 Python 3.10.21, PyTorch 2.14.0+cpu, Gymnasium 1.3.0, Stable-Baselines3/sb3-contrib 2.9.0입니다. 요구사항 파일은 최소 버전 범위를 사용하므로 새 설치가 기존 환경과 완전히 동일하다는 보장은 없습니다.

### 2. 검증과 실험 실행

```powershell
# 기존 PBRS·관측 래퍼 검증 (저장된 실험 결과를 덮어쓰지 않음)
python -c "from src.pbrs_unit_tests import run_pbrs_unit_tests; print(run_pbrs_unit_tests())"
python -c "from src.obs_wrapper_unit_tests import run_obs_wrapper_unit_tests; print(run_obs_wrapper_unit_tests())"

# 그래프·pointer 인터페이스 확인
python -m src.run_graph_interface_smoke

# 학습 없이 Rule 상한 평가
python -m src.run_rule_ceiling
```

`run_*` 모듈은 같은 이름의 `reports/`·`results/` 파일을 덮어쓸 수 있으며, 학습 모듈은 체크포인트도 갱신합니다. 기존 기록과 비교할 때는 Git 커밋과 체크포인트를 고정하고 실행 전후 변경을 확인하세요.

학습 실험은 다음 순서로 실행할 수 있습니다. 전체 학습은 시간이 걸립니다.

```powershell
python -m src.run_bc_rule
python -m src.run_rule_v2_ceiling
python -m src.run_graph_expert_bc
python -m src.run_p1_commit_fix
python -m src.run_p1_commit_ablation
```

학습·평가 설정은 [protocol.yaml](netheal-topology-generalization/config/protocol.yaml), 평가 목록은 [eval_suite.yaml](netheal-topology-generalization/config/eval_suite.yaml)을 참고하세요. 실험마다 seed·평가 수·토폴로지가 다르므로 개별 모듈과 보고서의 조건을 함께 확인해야 합니다.

### 3. 별도 telemetry 연구 자료 받기

```powershell
# 저장소 루트에서 실행: 대용량 원본 데이터가 포함됩니다.
git submodule update --init OutlierDenStream-BigDama18 telemetry
```

DenStream 참고 구현의 실행법은 해당 원본 README를 따릅니다. NetHeal용 Python 환경에서의 호환성은 별도로 검증하지 않았습니다.

## 문서와 다음 단계

먼저 [문서 인덱스](docs/README.md)에서 전체 취합 보고서와 세부 설계서를 확인하세요. 기존 요약에는 과거 실행 수치가 포함될 수 있으므로 **개별 `results/`와 `reports/`의 조건별 기록을 우선**합니다.

다음 연구 과제는 관측에서 유도한 missing-incident/asymmetry 특징을 통한 Misconfiguration 대상 선택 개선, Generic commit 효과의 다중 seed 검증, Star+Linear 학습의 Mesh 전이 평가입니다. 이는 계획이며 완료된 결과가 아닙니다.

## 출처와 라이선스

NetHeal, OutlierDenStream, Cisco telemetry는 각각 원본 저장소의 저작권·라이선스 조건을 따릅니다. [외부 저장소 안내](docs/external-repositories.md)에 출처와 고정 커밋을 기록했습니다. 자체 연구 코드에는 별도 공개 라이선스를 지정하지 않았습니다.
