# 실행 스크립트 안내

2026-10-12 기준. 파일 위치는 import와 동결 소스 검증을 위해 유지한다. 아래는 찾기 위한 안내이며 모든 파일을 순서대로 실행하라는 뜻이 아니다. 실행 전 각 명세의 출력 경로와 재개 조건을 확인한다.

## Jev 공식 평가 — 완료

| 역할 | 코드 |
|---|---|
| 최종 v4 평가 실행 | [jev_official_eval_v4.py](jev_official_eval_v4.py) |
| 손실 없는 입력 압축 | [jev_context_codec_v4.py](jev_context_codec_v4.py) |
| 긴 JSON 트리 분할 | [jev_tree_partition_v4.py](jev_tree_partition_v4.py) |
| 최종 결과 보고·검증 | [report_jev_official_eval_v4.py](report_jev_official_eval_v4.py), [audit_jev_partitions_v4.py](audit_jev_partitions_v4.py) |
| 분할 점검 | [check_jev_partition_v4.py](check_jev_partition_v4.py) |

최종 [보고서](../../docs/research/official_benchmark/reports/netheal_official_jev_context_fix_20261012.md)와 [v4 명세](../../docs/specs/netheal_official_jev_execution_v4_20261011.md)를 우선한다. 버전 없는 Jev 파일과 v2/v3 파일은 과거 실행·개발 기록용으로 보존한다. v4의 실행·codec·tree 소스는 동결되어 있으며 변경하지 않는다. 재측정은 유료 호출을 수반할 수 있다.

## LAYA 공식 환경 — 새 세션 재개 대상

| 단계 | 코드 | 현재 상태 |
|---|---|---|
| 짧은 입력 8-bit 전체 Encoder 시험 | [laya_official_resource_probe.py](laya_official_resource_probe.py) | 8 updates 및 복원 검증 완료 |
| 긴 입력 자원 시험 | [laya_official_long_resource_probe.py](laya_official_long_resource_probe.py) | 8 updates 완료, reserved 메모리가 물리 VRAM 초과 |
| 학습·개발 데이터 생성 | [laya_official_data.py](laya_official_data.py) | train 40 / dev 20 생성 완료 |
| 공개 입력 구성·분할 | [laya_official_input.py](laya_official_input.py) | 구현, 본 학습 미실행 |
| 학습 질문 준비 | [laya_official_prepare.py](laya_official_prepare.py) | 중단, 완료 산출물 없음 |
| 메모리 상한 시험 | [laya_official_capped_probe.py](laya_official_capped_probe.py) | 미실행 |

본 학습 루프는 아직 구현되지 않았다. [인계 문서](../../docs/handoffs/2026-10-12_laya_handoff.md) 및 [학습 명세](../../docs/specs/laya_official_training_20261012.md)를 읽고 재개한다. 기존 빈 prepared 디렉터리 처리 등 재개 주의점이 있다.

## 이전 LAYA 대조 실험

| 실험 | 실행·검증·보고 코드 이름 |
|---|---|
| 학습 가능 여부·gradient 감사 | `laya_learning_controls*.py`, `laya_gradient_audit.py`, `report_laya_learning_controls.py` |
| 짧은 신호와 기호 대조 | `laya_short_controls.py`, `laya_symbol_controls.py`, `report_laya_short_controls.py`, `report_laya_symbol_controls.py` |
| 학습률·예산·순서 대조 | `laya_optimization_controls.py`, `verify_laya_optimization_restore.py`, `report_laya_optimization_controls.py` |
| 실제 관측 표·문장 대조 | `laya_observation_*.py`, `verify_laya_observation_data.py`, `report_laya_observation_controls.py` |
| 초기 진단 pilot | `evaluate_laya_diagnosis_only.py`, `verify_laya_checkpoint.py`, `report_laya_pilot.py` |

보고서와 설계서는 [문서 인덱스](../../docs/README.md), 실제 산출물은 [결과 목록](../results/README.md)에서 찾는다. 과거 스크립트에는 고정 출력 경로가 있으므로 탐색 목적으로 실행하지 않는다.
