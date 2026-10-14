# Jev 본 평가 실행 조건 동결

일자: 2026-10-11 (KST). 사용자가 Jev 유료 API 호출을 명시적으로 허용했다.

원본: Claude Run `685bad60-d554-4300-9c7e-e849301d6df7`, 공개 스냅샷 100개, 고정 green 이미지 소스. fingerprint 100/100 일치, 원본 코드의 도구 8종과 최종 채점 사전 검증 통과. 원본 가중 성공률 65.0376%, 단순 성공률 66%.

개발 smoke: 평가와 별도인 seed 93001101에서 3 solver turn, Jev 1.13.0 요청 5회 모두 HTTP 200. scan_network/discover_neighbors/ping 실행 확인. 목표 정확도를 확인하는 실험이 아니라 연결 검사다. Windows cp949 읽기 오류는 API 요청 전 발생했고 `-X utf8` 실행으로 해결했다.

## 고정 조건

- 모델 `jev-1.13.0`, direct TypeSafe `/v1/systemone`, 출력은 choice만 사용.
- 원본 task/objective/hint와 MCP 도구 스키마 및 모든 도구 결과 히스토리를 제공한다. 생성형 설명 요구 부분은 제거한다.
- 도구 선택 후 인자를 고른다. 진단은 예측 유형→위치 순서. 정답과 숨겨진 상태는 모델에 전달하지 않는다.
- 인자 선택 전에 원본 무료 list_actions를 자동 호출하고 결과를 히스토리에 남긴다. 원본 solver와의 명시적 차이다.
- 최대 100 solver turn, 스냅샷 환경 100 action, episode당 1200초, 동시성 5. 요청 시작 간격 최소 0.5초로 공개 rate limit 부담을 낮춘다. 이는 지연 측정에 포함된다.
- 모델 요청당 최대 120초. 네트워크·일시적 서버 오류는 전체 episode 최대 10회 재시도하며 이전 시도 로그를 보존한다. 오진·정상 종료·단계 제한은 재시도하지 않는다.
- HTTP 400/413/422 등 결정적 입력 오류는 같은 유료 궤적을 반복하지 않고 오류로 종료한다. 원본의 모든 error 재시도 규칙과 차이로 기록한다.
- 관측 히스토리를 자동 절단하거나 다른 LLM으로 요약하지 않는다. Jev 입력 한도 초과는 별도의 실행 실패로 남기고, 진단 오답과 구분한다. 본 평가를 보고 표현/정책을 바꾸지 않는다.
- 공식 집계는 원본 CompetitionEvaluator를 사용한다. 별도로 모든 100개 사례를 분모로 하는 성공률과 오류/미완료 수를 보고한다. 원본은 오류의 metrics를 누락할 수 있으므로 오류가 있으면 원본식 집계와 전체 분모 집계를 구분한다.
- 동일 사례 원본 진단 결과와 paired 비교 가능. 단, 서로 다른 solver와 native Windows 직접 handler 실행 차이가 남으므로 순수 모델 우열이나 동일 컨테이너 재현이라고 쓰지 않는다.

## 비용과 한도

[TypeSafe 공식 모델 문서](https://docs.typesafe.ai/models), 2026-10-11 확인: Jev 1.13 입력 100만 token당 $0.042, 출력 무료. 단일 질문은 state+질문 32K, 전체 요청 64K 한도다.

연결 방식은 solver turn당 최대 3 API 호출, 기본 100개 평가의 최대 30,000회다. 각 요청 32K 기준 단순 상한 추정은 약 $40.32이며, 최대 11시도 전부 실행하는 극단적 경우 약 $443.52다. 실제 비용은 종료 시점과 usage로 계산하며 청구서 확정액이라고 쓰지 않는다. 입력 오류는 재시도하지 않는다. smoke 사용량은 37,385 input tokens로 약 $0.00157다.

## 재현 및 결과 보존

저장소 루트에서:

```powershell
.runtime/jev-official-venv/Scripts/python.exe -X utf8 netheal-topology-generalization/scripts/jev_official_eval.py evaluate
```

본 평가 전 결과 폴더에 frozen_protocol.json과 실행 소스 사본/해시를 기록한다. 기존 완료 episode는 재실행하지 않으며 중간 시도가 남으면 임의 덮어쓰기하지 않는다. 결과 위치: `netheal-topology-generalization/results/netheal_official_jev_20261011_v1/`.
