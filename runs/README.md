# 실행 산출물

| 폴더 | 실험 |
|---|---|
| [generalization](generalization/) | reliability v2, factorial v3 |
| [api_pilot](api_pilot/) | 공식 환경 이전 Sol·Jev 예비 비교 |
| [laya_pilot](laya_pilot/) | 초기 LAYA baseline·head 학습·진단·자원 점검 |
| [laya_controls](laya_controls/) | 학습 가능 여부·신호·예산·관측 표현 대조 |
| [official_benchmark/jev](official_benchmark/jev/) | 공식 Jev v2/v3/v4. 최종 v4만 100개 완료·문맥 오류 0 |
| [official_benchmark/laya/resource_probes](official_benchmark/laya/resource_probes/) | 전체 Encoder 8-bit 자원 시험 2개. 각 8 updates, 본 학습 아님 |
| [legacy_poc](legacy_poc/) | 이전 PoC 산출물 |
| [archive](archive/) | 임시 smoke·디버그 보관 |

실제 파일은 이곳에 있다. 기존 `netheal-topology-generalization/results/` 경로는 연결을 통해 계속 접근할 수 있다. 폴더를 복제한 것이 아니므로 용량을 이중 합산하지 않는다. 원자료·저장 가중치·동결 소스 내용은 변경하지 않았다.

진행 중인 [LAYA 데이터/준비](../netheal-topology-generalization/results/laya_official_20261012_v1/)와 [원본 공식 환경](../netheal-topology-generalization/results/netheal_official_jev_20261011_v1/)은 의도적으로 기존 위치에 보존했다. 이전 보고서 내부의 실행 경로 문자열은 역사적 기록이다.

새 실행은 `YYYY-MM-DD_<조건>_rNN/` 형식으로 분리한다. 현재 결과를 덮어쓰지 않는다. 보고서 요약은 [docs/research](../docs/research/)를 확인한다.
