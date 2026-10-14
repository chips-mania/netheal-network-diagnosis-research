# NetHeal Network Diagnosis Research

포트폴리오에서는 코드·논문·결과 요약을 제공하며, 대용량 가중치와 원본 실행 로그는 로컬에 보존합니다. [공개 범위와 재현 조건](docs/portfolio_publication.md)을 확인하세요.

- **2026-10-13 JEV 구조화 상태 실험 완료:** 원본 로그 보존, 입력 분할·판단 점수 가중평균 없음. 같은 100개 사례에서 JEV **48/100**, 기존 공개 Claude **66/100**. JEV 입력/API 오류 0건, 실제 요청 1,946건의 원본 로그 기반 재구성 검증 통과. Claude는 재실행하지 않아 입력 형식은 다릅니다. [결과 보고서](docs/research/official_benchmark/reports/jev_structured_state_20261013.md).

## 현재 작업 상태 (2026-10-12)

- **Jev 공식 사례 재측정 완료:** 100개 중 성공 42 / 오진 58 / 문맥 한도 미완료 0. [최종 보고서](docs/research/official_benchmark/reports/netheal_official_jev_context_fix_20261012.md). 원본 Claude는 66/100이며 실행 방식 차이는 보고서에 명시했습니다.
- **LAYA 후속 작업은 새 세션에서 재개:** 8비트 optimizer 자원 시험 및 학습 40·개발 20개 데이터 생성 완료. 입력 준비 미완료, 본 학습·공식 평가 미실행. [인계 문서](docs/handoffs/2026-10-12_laya_handoff.md).
- **찾아보기:** [새 구조·호환성 안내](docs/architecture/README.md) · [실험 실행 진입점](experiments/README.md) · [주제별 원시 결과](runs/README.md) · [전체 문서](docs/README.md).
- **디스크:** 재구성 시작 시 공간 부족이 있었으나 최종 확인에서는 C: 약 9.4GB 여유가 있었습니다. LAYA 본 학습 직전에 다시 확인하세요.

아래의 과거 실험 수치는 각기 다른 조건의 기록입니다. 공식 사례 재측정과 합쳐 비교하지 않습니다.

네트워크 장애 진단 에이전트가 **도구로 증거를 수집하고 장애 유형·위치를 판단하는 절차를 학습할 수 있는지**, 그리고 학습한 절차가 **다른 네트워크 크기와 토폴로지로 전이되는지** 검증하는 연구 저장소입니다.

## 목적

NetHeal 시뮬레이터 위에서 강화학습(PPO), 전문가 궤적 모방학습(Behavior Cloning, BC), GNN+pointer 정책을 비교합니다. 단순 성공률뿐 아니라 도구 사용, 성급한 진단, 타임아웃, 증거가 충분한데도 진단을 제출하지 않는 현상까지 분석합니다.

현재 중심 연구는 **네트워크 장애 진단 정책의 실패 모드와 크기 일반화**입니다. Cisco telemetry 기반 정상 패턴 전이 및 분포 변화 대응은 별도 연구 계획으로 보존되어 있으며, 이 저장소의 NetHeal 실험 성과와 구분합니다.

## 내용

- **환경 감사:** 관측·보상·액션 마스크를 점검하고 PPO가 `scan → diagnose`에 머무르는 원인을 분석합니다.
- **학습 프로토콜 비교:** PBRS, 타임아웃 패널티, 탐색, 진단 게이트, 관측 래퍼, Rule 기반 BC를 비교합니다.
- **일반화 실험:** Star 크기를 바꾸고, MLP와 GNN+pointer를 비교하며, Mesh 전이를 평가합니다.
- **병목 분해:** 진단 제출(commit)과 증거에 맞는 대상 노드 선택(pointer binding)을 분리해 분석합니다.

## API 비교 예비 실험 (2026-10-09)

GPT-6.1 Sol과 Jev를 같은 64개 사례에서 비교한 128회 pilot을 완료했습니다. Exact는 **87.5% / 26.6%**, 평균 진단 시간은 **21.07초 / 1.36초**, 사례당 usage 기반 추정 비용은 **$0.0480 / $0.00156**입니다. 개발용 공통 특징 표·프롬프트 비교이며 본 평가와 반복·이름 변경 검증은 아직 수행하지 않았습니다.

[결과와 프롬프트 설명](docs/research/api_pilot/reports/gpt_jev_pilot_v1.md) · [실험 설계](docs/specs/gpt_jev_comparison_v1.md)

## 최신 가설 검증 v3 (결과 검증: 2026-10-09)

사전 고정한 **ID·순서 처리 × 학습 구조의 3×2 실험**을 완료했습니다. 3개 replica, 18개 모델, 800개 독립 사례의 원래/변경 이름 평가와 Rule 기준을 합쳐 **29,600회 평가**했습니다. LAYA·Jev·일반 LLM은 제외했습니다.

- **H1 — ID·순서 증강:** 이름 변경 조건 Exact **+11.1%p**, 97.5% CI **[+6.2, +16.1]**.
- **H2 — 학습 구조 다양화:** 미학습 Mesh·Random Exact **+15.8%p**, 97.5% CI **[+11.3, +21.2]**.
- **남은 한계:** 혼합+증강도 원래 이름 Mesh 10 **12.7%**, Random 10 **7.0%**입니다. 정상 상태 오경보·복합 장애·실장비 일반화는 미검증입니다.
- **검증:** 관련 테스트 12개 통과, 소스·데이터·모델 해시와 평가 완전성 검증 통과.

[가설별 결론과 운영·네트워크 지표](docs/research/generalization/reports/reliability_v3_conclusions.md) · [전체 결과표](netheal-topology-generalization/results/factorial_v3_20261008/report.md) · [사전 설계](docs/specs/reliability_v3_hypotheses.md) · [실행 방법](docs/specs/reliability_v3_runbook.md)

## v2 탐색적 재실험 (2026-10-08)

내부 상태 입력을 제거하고 동일 데모·관측·평가 사례로 **3개 학습 seed, 35,200회 진단 평가**를 완료했습니다. 독립 평가 사례는 1,600개이며 같은 사례를 모델·seed별로 반복했습니다.

- **크기 전이:** GNN 기본 Star 8/10/12 Exact 평균 **90.8% / 60.7% / 51.0%**. 같은 입력의 새 MLP는 **19.8% / 14.3% / 4.8%**입니다.
- **개입 효과 재검토:** Star 10에서 추가 학습 대조군 **66.3%**, 가중치+샘플링 **64.8%**로 Generic의 일관된 우위는 확인하지 못했습니다. 샘플링만 적용한 결과는 **80.0%**지만 다른 크기의 안정성은 부족합니다.
- **미해결 일반화:** GNN 기본 Mesh 6은 **22.3%**. Generic의 Star 10은 순위 특징 교란 시 **64.8% → 27.8%**로 하락합니다.
- **다방면 지표:** 오진·미진단, 유형별 F1, 도구 호출·반복·비용, 진단 단계, CPU 추론 지연, 공통 all-pairs ping 실패·지연·변동·패킷 카운터를 기록했습니다. 실제 처리량·packet-level jitter·MTTR은 미측정입니다.

[핵심 결론과 다음 연구](docs/research/generalization/reports/reliability_v2_conclusions.md) · [전체 결과표](netheal-topology-generalization/results/reliability_v2_20261008/report.md) · [프로토콜](docs/specs/reliability_v2_protocol.md)

기존 PoC와 입력·데이터·예산이 달라 과거 숫자와 직접 개선 전후로 비교하지 않습니다. 아래는 과거 실험의 보관 기록입니다.

## 기존 PoC 결과 (보관)

아래 수치는 **이미 저장된 실험 산출물**의 요약이며, 이번 디렉터리 정리 과정에서 재학습한 결과가 아닙니다. Exact는 장애 **유형과 위치를 모두 맞힌 비율**입니다. 주로 단일 학습 seed와 조건별 100개 평가 에피소드로 수행한 PoC이므로 다중 seed 검증이나 실환경 성능을 뜻하지 않습니다.

- **Rule과 BC:** Star 6에서 GT를 행동 선택에 사용하지 않는 Rule은 Exact 100%. 현재 저장된 BC-rule 결과도 **100/100, 평균 5.86 step**입니다. 과거 취합 문서의 99%와 현재 결과 파일이 달라 여기서는 개별 결과 파일을 우선합니다. [Rule 보고서](netheal-topology-generalization/reports/rule_ceiling_poc.md) · [BC 보고서](netheal-topology-generalization/reports/bc_rule_poc.md) · [BC 원자료](netheal-topology-generalization/results/bc_rule_poc.json)
- **GNN의 부분적 크기 일반화:** Star {4,5,6} 전문가 데모로 학습한 P1 GNN+pointer는 Star 6/8/10에서 **100% / 90% / 42%**. 같은 비교의 B1 MLP는 **89% / 76% / 57%**입니다. Star 8에서는 +14%p지만 Star 10에서는 더 낮아 완전한 크기 일반화로 해석할 수 없습니다. [비교 보고서](netheal-topology-generalization/reports/graph_expert_poc.md)
- **진단 제출 병목 개선:** Generic commit supervision(진단 샘플 ×3, 진단 가중치 ×4)은 Star 10 Exact를 **42% → 66%**, Link 성공률을 **0% → 92%**로 개선했습니다. 다만 Star 8 Exact는 **90% → 86%**이고, 해당 평가의 Star 10 Misconfiguration은 **0%**로 남습니다. [개입·절제 보고서](netheal-topology-generalization/reports/p1_commit_ablation.md)
- **미해결 과제:** Star-only 학습 P1의 Mesh 6 Exact는 **9%**입니다. 토폴로지 계열 전이, 큰 Star에서의 Misconfiguration 대상 선택, 다중 seed 재현은 아직 해결되지 않았습니다. [전체 결과 취합](docs/research/generalization/reports/GNN_expert_guided_results_report.md)

연구의 현재 성과는 **관측된 PPO 실패 모드를 정리하고, 전문가 절차 모방의 상한과 일반화 병목을 분리한 것**입니다. 보상·마스크를 바꾼 실험은 공식 원본 조건과 다른 실험 변형으로 해석해야 합니다.

## 디렉터리 구조

```text
.
├── src/netheal_research/             # 공통 경로·실행기·환경·평가 코드
├── experiments/                     # 연구 주제별 실행 진입점 9개
├── configs/                         # 공유 설정 안내
├── tests/                           # 새 구조 검증; 기존 tests도 함께 실행
├── data/generated/laya/             # 기존 LAYA 학습 데이터
├── runs/                            # 실제 실험 산출물
│   ├── generalization/
│   ├── api_pilot/
│   ├── laya_pilot/
│   ├── laya_controls/
│   ├── official_benchmark/
│   ├── legacy_poc/
│   └── archive/
├── docs/
│   ├── architecture/                # 구조·이동 기록·검증
│   ├── research/                    # 주제별 보고서
│   ├── handoffs/                    # 새 세션 인계
│   └── specs/                       # 동결된 실행 전 명세
├── scripts/maintenance/             # 재구성·검증·호환 경로 복구
├── netheal-topology-generalization/ # 기존 실행 코드·호환 경로
├── netheal-ai-agent-benchmark/      # NetHeal submodule
├── OutlierDenStream-BigDama18/       # 참고 구현 submodule
├── telemetry/                      # 별도 연구 데이터 submodule
└── .runtime/                       # 기존 가상환경·모델 캐시
```

기존 결과 경로에는 연결 폴더를 남겨 실행 호환성을 유지합니다. 진행 중인 공식 LAYA 데이터와 공식 환경 원본은 기존 위치에 보존합니다. [구조·NTFS 연결·새 PC 복구 안내](docs/architecture/README.md)를 확인하세요. 연구용 원시 결과와 가중치는 삭제하지 않았습니다.

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

### v2 재실험 실행 및 검증

아래 명령은 저장소 루트에서 시작합니다. 새 실행은 기존 출력 폴더를 덮어쓰지 않습니다.

```powershell
$env:PYTHONPATH = (Resolve-Path ./netheal-ai-agent-benchmark).Path
cd netheal-topology-generalization
python -m pytest tests/test_research_suite.py -q
python -m src.run_research_suite --out results/reliability_v2_new
python -m src.verify_research_run results/reliability_v2_new
python -m src.report_research_suite results/reliability_v2_new
```

pytest가 없으면 개발 환경에 설치하세요. 이번 실행은 프로젝트의 `.runtime/venv`에서 진행했으며 정확한 패키지는 [환경 기록](netheal-topology-generalization/results/reliability_v2_20261008/environment-freeze.txt)에 있습니다. 보고된 실행 시점 소스와 달라지면 해시 검증은 의도적으로 실패합니다.

아래의 기존 `run_*` 명령은 과거 PoC용입니다. 현재 관측 코드가 보완되어 과거 수치의 그대로인 재현을 보장하지 않으며, 새 비교에는 위의 공통 실행기를 사용하세요.

### 2. 기존 검증과 실험 실행

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

최신 재실험에서 Generic 효과를 다중 seed로 재검토했습니다. 다음 과제는 데이터 수집 seed 반복, 순위 특징 의존 완화, 같은 총 데모 예산의 Star+Linear 학습과 Mesh 전이, 정상·복합 장애 및 실네트워크 측정, 일반 LLM/후속 모델의 공통 평가입니다.

## 출처와 라이선스

NetHeal, OutlierDenStream, Cisco telemetry는 각각 원본 저장소의 저작권·라이선스 조건을 따릅니다. [외부 저장소 안내](docs/external-repositories.md)에 출처와 고정 커밋을 기록했습니다. 자체 연구 코드에는 별도 공개 라이선스를 지정하지 않았습니다.


## LAYA 후속 실험

[LAYA 학습·진단 pilot](docs/research/laya_controls/reports/laya_pilot_v2.md)에서 실행 상태와 결과를 확인할 수 있습니다. [설계 및 재현 명령](docs/specs/laya_diagnosis_v1.md)에는 GPU 설정, 에피소드 분할, 학습 범위와 비교 제한을 기록했습니다.

- [LAYA 학습 가능 여부 검증 (2026-10-10)](docs/research/laya_controls/reports/laya_learning_controls_20261010.md)
- [LAYA 짧은 입력·부분 Encoder 학습 검증](docs/research/laya_controls/reports/laya_symbol_controls_v1.md)
- [LAYA 학습률·학습 예산·선택지 순서 검증](docs/research/laya_controls/reports/laya_optimization_controls_v1.md)
