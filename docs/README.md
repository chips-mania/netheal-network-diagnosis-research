# 연구 문서 인덱스

보고서는 [연구 주제별 디렉터리](research/)로, 인계 문서는 [handoffs](handoffs/)로 이동했습니다. 이전 문서 경로에도 안내를 남겼습니다. [새 구조와 호환성](architecture/README.md) · [실험 진입점](../experiments/README.md) · [실제 결과 위치](../runs/README.md)를 참고하세요.

## 최신 상태와 새 세션 시작점 (2026-10-12)

- [프로젝트 디렉터리·환경·보존 규칙](project_directory_20261012.md)
- [LAYA 새 세션 인계: 완료·중단·미실행 구분](handoffs/2026-10-12_laya_handoff.md)
- [원시 결과 전체 목록](../netheal-topology-generalization/results/README.md)
- [실행·검증·보고 스크립트 안내](../netheal-topology-generalization/scripts/README.md)

## 공식 NetHeal 사례 — Jev 완료, LAYA 준비

- **최종 결과:** [Jev 문맥 한도 수정 후 100개 재측정](research/official_benchmark/reports/netheal_official_jev_context_fix_20261012.md)
- 이전 기록: [사전 점검](research/official_benchmark/reports/netheal_official_jev_preflight_20261011.md) · [문맥 수정 전 결과](research/official_benchmark/reports/netheal_official_jev_results_20261011.md)
- 공통 계획: [Jev·LAYA 공식 환경 평가 계획](specs/netheal_official_jev_laya_evaluation_20261011.md)
- Jev 실행 명세 이력: [v1](specs/netheal_official_jev_execution_20261011.md) · [v2](specs/netheal_official_jev_execution_v2_20261011.md) · [v3](specs/netheal_official_jev_execution_v3_20261011.md) · [v4](specs/netheal_official_jev_execution_v4_20261011.md) · [중단 후 재개](specs/netheal_official_jev_v4_resume_20261012.md)
- LAYA 사전 명세: [본 학습](specs/laya_official_training_20261012.md) · [입력 구성](specs/laya_official_input_execution_20261012.md) · [메모리 상한 시험](specs/laya_official_memory_cap_20261012.md)
- 완료된 LAYA 자원 시험의 명세: [짧은 입력](specs/laya_official_resource_probe_20261012.md) · [긴 입력](specs/laya_official_long_resource_probe_20261012.md). 각각 8 updates 시험이며 본 학습 완료를 뜻하지 않습니다.

목적과 주요 결과는 [프로젝트 README](../README.md)에서 시작하세요. 원래 루트에 있던 문서를 주제별로 분류했으며, 아래의 원본 파일명과 내용은 보존했습니다. 기존 문서의 코드 경로는 별도 언급이 없으면 저장소 루트 기준이며, 문서명만 적힌 과거 참조는 이 인덱스에서 찾을 수 있습니다.

## 완료된 가설 실험 v3

- [가설별 결론과 남은 공백](research/generalization/reports/reliability_v3_conclusions.md)
- [전체 결과표](../netheal-topology-generalization/results/factorial_v3_20261008/report.md)

- [실행 전 가설·3×2 설계](specs/reliability_v3_hypotheses.md)
- [실행 및 완료 여부 확인](specs/reliability_v3_runbook.md)

## v2 탐색적 재실험

- [신뢰도 보완 재실험 결론](research/generalization/reports/reliability_v2_conclusions.md)
- [실험 프로토콜과 네트워크 지표 정의](specs/reliability_v2_protocol.md)
- [완료된 전체 결과표](../netheal-topology-generalization/results/reliability_v2_20261008/report.md)

## API 예비 실험

- [GPT·Jev pilot 결과와 프롬프트 설명](research/api_pilot/reports/gpt_jev_pilot_v1.md)

- [GPT-6.1 Sol와 Jev 비교 설계·과금·시간·정확도](specs/gpt_jev_comparison_v1.md)

## 연구 계획

- [netheal_topology_generalization_research_plan.md](plans/netheal_topology_generalization_research_plan.md)
- [network_anomaly_drift_maintenance_plan.md](plans/network_anomaly_drift_maintenance_plan.md)
- [network_anomaly_transfer_experiment_plan.md](plans/network_anomaly_transfer_experiment_plan.md)

## 실험 설계 (수정본 우선)

- [Ablation_NL_spec_수정.md](specs/Ablation_NL_spec_수정.md)
- [E1_PBRS_spec_수정.md](specs/E1_PBRS_spec_수정.md)
- [E2_timeout_spec.md](specs/E2_timeout_spec.md)
- [E3_exploration_spec.md](specs/E3_exploration_spec.md)
- [GNN_expert_guided_spec.md](specs/GNN_expert_guided_spec.md)
- [Obs_then_E4_spec.md](specs/Obs_then_E4_spec.md)

## 결과 취합과 이력

- [GNN_expert_guided_results_report.md](research/generalization/reports/GNN_expert_guided_results_report.md)
- [NetHeal_experiment_history.md](research/generalization/reports/NetHeal_experiment_history.md)
- [NetHeal_RL_results_summary.md](research/generalization/reports/NetHeal_RL_results_summary.md)

## 수정 전 설계서 보관

- [Ablation_NL_spec.md](archive/specs/Ablation_NL_spec.md)
- [E1_PBRS_spec.md](archive/specs/E1_PBRS_spec.md)

## 수치 해석

- `results/` 아래 취합 문서는 당시 실행 결과와 해석을 담은 역사적 기록입니다.
- [개별 보고서](../netheal-topology-generalization/reports)와 [구조화 결과](../netheal-topology-generalization/results)의 실행 조건 및 현재 저장 수치를 먼저 확인하세요.
- BC-rule 과거 취합에는 99%가 남아 있으나 현재 개별 보고서·JSON은 100%입니다. 새 README는 현재 파일을 따릅니다.
- GNN Generic commit의 Misconfiguration 수치도 평가마다 다릅니다. commit ablation의 Star10은 0%, 별도 path audit 취합은 8%로 기록되어 있어 서로 동일 평가로 합치지 않습니다.
- [외부 저장소 출처와 버전](external-repositories.md)


## LAYA 후속 실험

- [실제 관측 표·문장 대조 결과](research/laya_controls/reports/laya_observation_controls_v1.md) · [설계](specs/laya_observation_controls_v1.md) · [실행 조건](specs/laya_observation_controls_v1_execution.md)

- [학습·진단 pilot 실행 상태와 결과](research/laya_controls/reports/laya_pilot_v2.md)
- [설계 및 재현 명령](specs/laya_diagnosis_v1.md)

- [LAYA 학습 가능 여부 검증 (2026-10-10)](research/laya_controls/reports/laya_learning_controls_20261010.md)
- [LAYA 짧은 입력·부분 Encoder 학습 검증](research/laya_controls/reports/laya_symbol_controls_v1.md)
- [LAYA 학습률·학습 예산·선택지 순서 검증](research/laya_controls/reports/laya_optimization_controls_v1.md)
