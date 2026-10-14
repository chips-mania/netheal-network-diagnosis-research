# 구조 변경 원문 Jev: 원본 Claude 기준선 대비 첫 사례 테스트

2026-10-13. 연구자가 공개한 Claude 결과를 기준선으로 재사용하고 Jev만 새로 실행했다. 사례 ID `00bd290a-df64-46a6-8f06-f087bde022a9`, seed 1283, 초기 scenario fingerprint 일치를 확인했다. 전체 100개 평가가 아닌 첫 사례 단독 smoke다.

## 조건

기존 v4 환경 루프·공개 도구·후보 생성·100단계/100 turn·1200초 예산·원본 채점기를 유지했다. 입력은 전체 공개 관측 이력을 포함하는 일반 JSON 문자열이며 자체 압축, 분할, 24KB 가드, 고정 요청 대기를 사용하지 않았다. 도구 정의는 name/description/inputSchema를 유지하고 스키마 title 주석을 제거했다.

원본 Claude purple 이미지에서 검증한 SYSTEM_PROMPT를 그대로 state에 포함했다. 과제·목표·힌트·예산은 공개 state 필드로 전달하고 초기 user 메시지 형식도 원본 코드에 맞췄다. 다만 Jev는 설명문과 native tool call을 생성하지 않으므로 해당 출력 요구를 선택형 decision으로 대체한다는 어댑터 지시를 명시했다. 도구 선택과 인자 선택은 별도 요청이며 후보는 공개 list_actions에서 얻는다. Claude와 프롬프트 바이트·메시지 형식·실행 환경이 완전히 같다는 뜻은 아니다. 과거 Claude 관측을 재생하지 않고 Jev가 실제로 선택한 도구 결과를 누적했다.

## 결과

| 지표 | 연구자 공개 Claude | 이번 Jev |
|---|---:|---:|
| 최종 진단 | link_failure / center->edge_5 | link_failure / center->edge_5 |
| 성공 | 예 | 예 |
| 환경 단계 | 8 | 7 |
| 에피소드 시간 | 79.159초 | 4.055초 |
| 모델 API 요청 | 원시 로그 미확보 | 14 |
| API 응답 시간 합계 | 미확보 | 3.511초 |
| 입력 토큰 합계 | 미확보 | 155,577 |
| 최대 요청 보고 토큰 | 미확보 | 19,552 |
| 추정 API 비용 | 미확보 | $0.006534 |

Jev 도구 순서는 scan_network → check_status(center) → discover_neighbors(center) → ping(center, edge_0) → traceroute(center, edge_1) → check_interfaces(center) → submit_diagnosis였다. 14개 API 요청 모두 성공했다. 입력 준비 합계는 0.028초였다. Jev 에피소드 시간에는 HTTP 클라이언트 준비와 도구 실행 등이 포함되지만 Python 시작·모듈 import·스냅샷 로딩은 포함되지 않는다.

단일 사례에서 원문 입력으로 진단 완료 가능함을 확인했다. 이 사례의 결과로 원문 방식의 정확도 개선, 다른 모든 사례의 한도 수용, 모델 자체 속도 배수 또는 Claude 대비 비용 절감률을 주장하지 않는다. Claude는 과거 컨테이너 실행, Jev는 현재 Windows 직접 handler 실행이며 선택 경로도 다르다. 입력 구성과 지시도 이전 압축 실험과 함께 바뀌었으므로 성능 차이를 압축 제거 하나의 효과로 해석할 수 없다.

## 산출물

- 실행: `experiments/official_benchmark/jev/readable_baseline_smoke.py`
- 설정·원본 시스템 프롬프트·소스 해시: `runs/official_benchmark/jev/2026-10-13_readable_baseline_smoke_r01/protocol.json`
- 비교: 같은 폴더 `comparison.json`
- 실제 요청·응답·도구 결과: 같은 폴더 `episode/`
- Claude 원본: `netheal-topology-generalization/results/netheal_official_jev_20261011_v1/upstream/manikyabard-20260201-100940.json`
