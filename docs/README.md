# 연구 문서 인덱스

목적과 주요 결과는 [프로젝트 README](../README.md)에서 시작하세요. 원래 루트에 있던 문서를 주제별로 분류했으며, 아래의 원본 파일명과 내용은 보존했습니다. 기존 문서의 코드 경로는 별도 언급이 없으면 저장소 루트 기준이며, 문서명만 적힌 과거 참조는 이 인덱스에서 찾을 수 있습니다.

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

- [GNN_expert_guided_results_report.md](results/GNN_expert_guided_results_report.md)
- [NetHeal_experiment_history.md](results/NetHeal_experiment_history.md)
- [NetHeal_RL_results_summary.md](results/NetHeal_RL_results_summary.md)

## 수정 전 설계서 보관

- [Ablation_NL_spec.md](archive/specs/Ablation_NL_spec.md)
- [E1_PBRS_spec.md](archive/specs/E1_PBRS_spec.md)

## 수치 해석

- `results/` 아래 취합 문서는 당시 실행 결과와 해석을 담은 역사적 기록입니다.
- [개별 보고서](../netheal-topology-generalization/reports/)와 [구조화 결과](../netheal-topology-generalization/results/)의 실행 조건 및 현재 저장 수치를 먼저 확인하세요.
- BC-rule 과거 취합에는 99%가 남아 있으나 현재 개별 보고서·JSON은 100%입니다. 새 README는 현재 파일을 따릅니다.
- GNN Generic commit의 Misconfiguration 수치도 평가마다 다릅니다. commit ablation의 Star10은 0%, 별도 path audit 취합은 8%로 기록되어 있어 서로 동일 평가로 합치지 않습니다.
- [외부 저장소 출처와 버전](external-repositories.md)
