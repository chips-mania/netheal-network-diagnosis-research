# 원시 결과 디렉터리 목록

작성일: 2026-10-12 KST. 기존 결과 경로를 유지한다. 용량은 하위 파일 논리 크기 합계이며 실행 완료 여부를 의미하지 않는다.

재구성 이후 대부분의 실제 산출물은 [runs](../../runs/README.md)와 [data](../../data/README.md)에 있습니다. 여기의 이전 경로는 연결로 유지합니다. 아래 용량 표는 재구성 전 스냅샷이며 새 경로와 합산하면 중복 집계됩니다.

## 최신 결과와 상태

| 경로 | 상태·역할 |
|---|---|
| [netheal_official_jev_20261011_v4](netheal_official_jev_20261011_v4/) | 최종 100개 완료: 성공 42 / 오진 58 / 문맥 오류 0. verification.json 검증 완료 |
| [netheal_official_jev_20261011_v3](netheal_official_jev_20261011_v3/) | 14개 결과 기록 후 중단. 최종 평가로 사용하지 않음 |
| [netheal_official_jev_20261011_v2](netheal_official_jev_20261011_v2/) | 이전 평가: 성공 34 / 오진 27 / 문맥 실패 39 |
| [netheal_official_jev_20261011_v1](netheal_official_jev_20261011_v1/) | 원본 환경 소스·평가 스냅샷·provenance 보관. 삭제·이동 금지 |
| [laya_official_20261012_v1](laya_official_20261012_v1/) | 학습 40·개발 20개 생성 완료. 입력 준비 미완료, 본 학습·공식 평가 미실행 |
| [laya_official_resource_20261012_v1](laya_official_resource_20261012_v1/) | 짧은 입력 전체 Encoder 8-bit 자원 시험 8 updates, 복원 검증 완료 |
| [laya_official_resource_long_20261012_v1](laya_official_resource_long_20261012_v1/) | 긴 입력 자원 시험 8 updates 완료. 예약 메모리 물리 VRAM 초과 기록 |

[Jev 최종 보고서](../../docs/results/netheal_official_jev_context_fix_20261012.md) · [LAYA 인계 문서](../../docs/specs/laya_handoff_20261012.md) · [전체 연구 문서](../../docs/README.md)

## 이전 실험군

- `factorial_v3_*`, `reliability_v2_*`: GNN·정책 일반화 및 신뢰도 실험.
- `api_*`: 공식 환경 재측정 이전의 Sol·Jev 예비 비교·점검.
- `laya_observation_controls_v1`: 실제 관측 표·문장 판단 대조.
- `laya_optimization_controls_v1`, `laya_symbol_controls_v1`, `laya_short_controls_v1`, `laya_learning_controls*`: 학습 예산·선택지 순서·신호 학습 대조.
- 나머지 `laya_*`: 초기 데이터·baseline·head 학습·자원 점검 기록. 일부 실패·중단 포함, 보고서의 개별 상태 확인.
- 루트 JSON·CSV 및 기타 디렉터리: 과거 PoC·평가 원자료. 정리 과정에서 재실행하거나 수치를 갱신하지 않음.
- `_tmp_*`와 빈 디렉터리도 보존. 이름이나 파일 수만으로 실험 완료·삭제 가능 여부를 판단하지 않음.

## 전체 산출물 목록

임시 smoke 2개와 디버그 로그 2개는 [archive](archive/README.md)로 이동했다. 아래 링크는 새 위치를 가리키며 CSV는 이동 전 재고 스냅샷으로 보존한다.

[기계 판독용 CSV](directory_inventory_20261012.csv). 이 안내 파일과 CSV 자체는 목록 집계에서 제외했다. 새 실험이 추가되면 이 스냅샷과 실제 디렉터리는 달라질 수 있다.

| 경로 | 종류 | 파일 수 | MiB |
|---|---|---:|---:|
| [_tmp_factorial_v3_smoke](archive/cleanup_20261012/smoke/_tmp_factorial_v3_smoke/) | directory | 38 | 4.63 |
| [_tmp_reliability_smoke](archive/cleanup_20261012/smoke/_tmp_reliability_smoke/) | directory | 24 | 4.92 |
| [api_pilot_20261009](api_pilot_20261009/) | directory | 13 | 11.67 |
| [api_preflight_20261009_041051](api_preflight_20261009_041051/) | directory | 3 | 0.01 |
| [api_smoke_20261009](api_smoke_20261009/) | directory | 12 | 0.20 |
| [factorial_v3_20261008](factorial_v3_20261008/) | directory | 83 | 113.99 |
| [laya_baseline_compact_fp32_v2](laya_baseline_compact_fp32_v2/) | directory | 108 | 4.80 |
| [laya_baseline_compact_v2](laya_baseline_compact_v2/) | directory | 4 | 0.09 |
| [laya_baseline_cpu_v1](laya_baseline_cpu_v1/) | directory | 0 | 0.00 |
| [laya_baseline_cpu_v1_retry](laya_baseline_cpu_v1_retry/) | directory | 2 | 0.02 |
| [laya_baseline_gpu_pilot_v1](laya_baseline_gpu_pilot_v1/) | directory | 2 | 0.03 |
| [laya_data_compact_v2](laya_data_compact_v2/) | directory | 6 | 24.24 |
| [laya_data_v1](laya_data_v1/) | directory | 5 | 45.69 |
| [laya_diagnosis_only_v2](laya_diagnosis_only_v2/) | directory | 3 | 0.31 |
| [laya_gpu_full_smoke_v1](laya_gpu_full_smoke_v1/) | directory | 3 | 101.15 |
| [laya_gpu_smoke_v1](laya_gpu_smoke_v1/) | directory | 3 | 101.15 |
| [laya_head_compact_48ep_v2](laya_head_compact_48ep_v2/) | directory | 9 | 807.00 |
| [laya_head_compact_fp32_v2](laya_head_compact_fp32_v2/) | directory | 108 | 4.80 |
| [laya_learning_controls_20261010](laya_learning_controls_20261010/) | directory | 34 | 607.14 |
| [laya_learning_controls_ce_20261010](laya_learning_controls_ce_20261010/) | directory | 17 | 303.57 |
| [laya_observation_controls_v1](laya_observation_controls_v1/) | directory | 172 | 898.78 |
| [laya_official_20261012_v1](laya_official_20261012_v1/) | directory | 125 | 35.96 |
| [laya_official_resource_20261012_v1](laya_official_resource_20261012_v1/) | directory | 9 | 2423.74 |
| [laya_official_resource_long_20261012_v1](laya_official_resource_long_20261012_v1/) | directory | 9 | 2423.74 |
| [laya_optimization_controls_v1](laya_optimization_controls_v1/) | directory | 67 | 1776.23 |
| [laya_pilot_report_v2](laya_pilot_report_v2/) | directory | 1 | 0.01 |
| [laya_short_controls_v1](laya_short_controls_v1/) | directory | 19 | 303.59 |
| [laya_symbol_controls_v1](laya_symbol_controls_v1/) | directory | 36 | 607.20 |
| [netheal_official_jev_20261011_v1](netheal_official_jev_20261011_v1/) | directory | 300 | 8.92 |
| [netheal_official_jev_20261011_v2](netheal_official_jev_20261011_v2/) | directory | 616 | 77.52 |
| [netheal_official_jev_20261011_v3](netheal_official_jev_20261011_v3/) | directory | 129 | 16.15 |
| [netheal_official_jev_20261011_v4](netheal_official_jev_20261011_v4/) | directory | 678 | 135.62 |
| [reliability_v2_20261008](reliability_v2_20261008/) | directory | 53 | 62.79 |
| [train_episode_log](train_episode_log/) | directory | 6 | 10.32 |
| [_n4_debug.txt](archive/cleanup_20261012/debug/_n4_debug.txt) | file | 1 | 0.00 |
| [ablation_nl_control_nl_det_trace.csv](ablation_nl_control_nl_det_trace.csv) | file | 1 | 0.02 |
| [ablation_nl_poc.json](ablation_nl_poc.json) | file | 1 | 0.01 |
| [ablation_nl_variant_4fault_det_trace.csv](ablation_nl_variant_4fault_det_trace.csv) | file | 1 | 0.02 |
| [ablation_nl_variant_nl_det_trace.csv](ablation_nl_variant_nl_det_trace.csv) | file | 1 | 0.02 |
| [bc_informed_poc.json](bc_informed_poc.json) | file | 1 | 0.04 |
| [bc_rule_gate_ppo_poc.json](bc_rule_gate_ppo_poc.json) | file | 1 | 0.11 |
| [bc_rule_poc.json](bc_rule_poc.json) | file | 1 | 0.04 |
| [bc_seed_gate_ppo_poc.json](bc_seed_gate_ppo_poc.json) | file | 1 | 0.07 |
| [device_failure_only_action_trace.csv](device_failure_only_action_trace.csv) | file | 1 | 0.04 |
| [diagnosis_action_mapping.json](diagnosis_action_mapping.json) | file | 1 | 0.01 |
| [diagnosis_factorization_audit.json](diagnosis_factorization_audit.json) | file | 1 | 0.01 |
| [diagnosis_gate_poc.json](diagnosis_gate_poc.json) | file | 1 | 0.00 |
| [diagnosis_gate_poc_control_trace.csv](diagnosis_gate_poc_control_trace.csv) | file | 1 | 0.02 |
| [diagnosis_gate_poc_variant_trace.csv](diagnosis_gate_poc_variant_trace.csv) | file | 1 | 0.03 |
| [e1_pbrs_poc.json](e1_pbrs_poc.json) | file | 1 | 0.01 |
| [e1_pbrs_poc_control_trace.csv](e1_pbrs_poc_control_trace.csv) | file | 1 | 0.02 |
| [e1_pbrs_poc_variant_trace.csv](e1_pbrs_poc_variant_trace.csv) | file | 1 | 0.02 |
| [e2_timeout_poc.json](e2_timeout_poc.json) | file | 1 | 0.00 |
| [e2_timeout_poc_control_trace.csv](e2_timeout_poc_control_trace.csv) | file | 1 | 0.02 |
| [e2_timeout_poc_variant_trace.csv](e2_timeout_poc_variant_trace.csv) | file | 1 | 0.02 |
| [e3_control_det_trace.csv](e3_control_det_trace.csv) | file | 1 | 0.02 |
| [e3_control_stoch_trace.csv](e3_control_stoch_trace.csv) | file | 1 | 0.03 |
| [e3_exploration_poc.json](e3_exploration_poc.json) | file | 1 | 0.01 |
| [e3_stage_a_probes.json](e3_stage_a_probes.json) | file | 1 | 0.00 |
| [e3_variant_det_trace.csv](e3_variant_det_trace.csv) | file | 1 | 0.02 |
| [e3_variant_stoch_trace.csv](e3_variant_stoch_trace.csv) | file | 1 | 0.03 |
| [evidence_diagnose_gate_poc.json](evidence_diagnose_gate_poc.json) | file | 1 | 0.07 |
| [factorized_diagnosis_poc.json](factorized_diagnosis_poc.json) | file | 1 | 0.00 |
| [factorized_diagnosis_poc_control_trace.csv](factorized_diagnosis_poc_control_trace.csv) | file | 1 | 0.02 |
| [factorized_diagnosis_poc_variant_trace.csv](factorized_diagnosis_poc_variant_trace.csv) | file | 1 | 0.16 |
| [failure_mode_audit.json](failure_mode_audit.json) | file | 1 | 0.01 |
| [graph_expert_bc_rerun.log](archive/cleanup_20261012/debug/graph_expert_bc_rerun.log) | file | 1 | 0.01 |
| [graph_expert_poc.json](graph_expert_poc.json) | file | 1 | 0.26 |
| [graph_interface_smoke.json](graph_interface_smoke.json) | file | 1 | 0.00 |
| [link_only_action_trace.csv](link_only_action_trace.csv) | file | 1 | 0.02 |
| [link_only_sanity_summary.json](link_only_sanity_summary.json) | file | 1 | 0.00 |
| [masked_ppo_action_trace.csv](masked_ppo_action_trace.csv) | file | 1 | 0.02 |
| [misconfiguration_only_action_trace.csv](misconfiguration_only_action_trace.csv) | file | 1 | 0.02 |
| [nonlink_single_fault_learnability.json](nonlink_single_fault_learnability.json) | file | 1 | 0.01 |
| [obs_then_e4_stage_b.json](obs_then_e4_stage_b.json) | file | 1 | 0.01 |
| [p1_commit_ablation.json](p1_commit_ablation.json) | file | 1 | 0.00 |
| [p1_commit_factor_ablation.json](p1_commit_factor_ablation.json) | file | 1 | 0.01 |
| [p1_commit_fix.json](p1_commit_fix.json) | file | 1 | 0.01 |
| [p1_commit_margin_audit.json](p1_commit_margin_audit.json) | file | 1 | 0.02 |
| [p1_misconfig_path_audit.json](p1_misconfig_path_audit.json) | file | 1 | 0.02 |
| [p1_misconfig_pointer_rank.json](p1_misconfig_pointer_rank.json) | file | 1 | 0.01 |
| [p1_pointer_curriculum.json](p1_pointer_curriculum.json) | file | 1 | 0.01 |
| [p1_pointer_only_probe.json](p1_pointer_only_probe.json) | file | 1 | 0.02 |
| [p1_star10_failure_audit.json](p1_star10_failure_audit.json) | file | 1 | 0.02 |
| [p1_train_audit.json](p1_train_audit.json) | file | 1 | 0.00 |
| [performance_degradation_only_action_trace.csv](performance_degradation_only_action_trace.csv) | file | 1 | 0.03 |
| [post_scan_audit.json](post_scan_audit.json) | file | 1 | 0.03 |
| [ppo_action_trace.csv](ppo_action_trace.csv) | file | 1 | 0.01 |
| [ppo_diagnosis_baselines.json](ppo_diagnosis_baselines.json) | file | 1 | 0.01 |
| [random_valid_baseline.csv](random_valid_baseline.csv) | file | 1 | 0.02 |
| [repeat_penalty_poc.json](repeat_penalty_poc.json) | file | 1 | 0.07 |
| [rule_ceiling_poc.json](rule_ceiling_poc.json) | file | 1 | 0.04 |
| [rule_v2_ceiling.json](rule_v2_ceiling.json) | file | 1 | 0.19 |
| [star_mix456_poc.json](star_mix456_poc.json) | file | 1 | 0.37 |
| [star_scale_poc.json](star_scale_poc.json) | file | 1 | 0.41 |
| [star_scale_replay_n4_extra.json](star_scale_replay_n4_extra.json) | file | 1 | 0.01 |
| [star_scale_replay_poc.json](star_scale_replay_poc.json) | file | 1 | 0.60 |
| [stochastic_evidence_audit.json](stochastic_evidence_audit.json) | file | 1 | 0.01 |
| [symmetric_reward_poc.json](symmetric_reward_poc.json) | file | 1 | 0.00 |
| [symmetric_reward_poc_control_trace.csv](symmetric_reward_poc_control_trace.csv) | file | 1 | 0.02 |
| [symmetric_reward_poc_variant_trace.csv](symmetric_reward_poc_variant_trace.csv) | file | 1 | 0.02 |
| [tool_solvability_audit.json](tool_solvability_audit.json) | file | 1 | 0.64 |

