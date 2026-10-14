# LAYA 첫 사례 시간 분해 — 2026-10-13

Jev 단독 계측과 동일한 사례 `00bd290a-df64-46a6-8f06-f087bde022a9`를 실제 도구와 로컬 LAYA로 실행했다. fingerprint는 두 실행 모두 `bd4ca296e5f4fce321c46157b4560a425cde64385e335faf8afa7edc098927e0`이다.

기존 LAYA 공식 평가 설정을 유지했다: 기본 checkpoint `7b928d828b7b0e022f929d9bd2e44165aa270148`, FP32, seed 1012, max_len/head_len 8192, GTX 1660 SUPER, 단일 사례, 환경 100단계·solver 100 turn·1200초 제한. 입력·분할·선택 규칙 변경 없이 계측 래퍼만 추가했다. 고정 대기는 없다.

## 결과

모델 준비 70.263초를 별도 측정했다. 준비 후 실제 사례 처리는 123.071초였다. 모델 forward 43회, 환경 도구 41회. 최종 제출은 `performance_degradation`, `center->edge_0`로 오답이다. 정답은 `link_failure`, `center->edge_5`다.

| 항목 | 초 | 비중 |
|---|---:|---:|
| 모델 forward 및 결과 반환 | 116.301 | 94.50% |
| 압축·복원 검사 | 5.652 | 4.59% |
| 토큰화·입력 길이 검사 | 0.464 | 0.38% |
| 나머지 분할 처리 | 0.301 | 0.24% |
| 로그 저장 | 0.158 | 0.13% |
| 환경 도구·후보 조회 | 0.0525 | 0.043% |
| 나머지 준비·어댑터 처리 | 0.144 | 0.12% |

중첩 구간은 배타적으로 계측했다. forward에는 배치 구성·GPU 전송·모델 계산·softmax·CPU로 결과 회수가 포함된다. CPU 회수로 GPU 결과 완료를 기다리므로 비동기 GPU 실행을 단순 제출 시간으로만 측정한 값이 아니다. Python 시작·초기 모듈 import·스냅샷 로딩은 별도 계측 밖이다. 모델 준비 시간에는 Backend 생성 내부 import와 checkpoint 로딩 등이 포함된다.

## 해석

Jev 단독 실행은 8.513초, 28회 API, 14단계로 정답이었다. LAYA는 같은 사례에서 scan_network를 반복했고, 입력 이력이 누적되며 판단 시간이 초기 약 1초에서 후반 약 5초까지 늘었다. LAYA의 이번 주된 병목은 모델 forward이며 고정 요청 대기나 시뮬레이터 도구가 아니다.

모델뿐 아니라 Jev 원격 서비스와 로컬 GTX 1660 SUPER FP32의 실행 환경, 토크나이저 및 실제 선택 경로가 다르다. 이 결과를 동일 입력·동일 하드웨어에서의 모델 속도 비교로 해석하면 안 된다. 단일 사례이므로 전체 성능 결론도 아니다. API 비용은 발생하지 않았으나 로컬 연산 비용이 무료라는 의미는 아니다.

## 자료

- 스크립트: `experiments/official_benchmark/laya/profile_first_case.py`
- 시간·자원·진단 집계: `runs/official_benchmark/laya/2026-10-13_first_case_latency_r01/profile.json`
- 실행 설정: 같은 폴더의 `protocol.json`
- 입력·응답·도구 전문: 같은 폴더의 `episode/`
