# Jev v2 실행 동결 — 2026-10-11 KST

기본 설정은 [v1 실행 명세](netheal_official_jev_execution_20261011.md)를 따른다. 다음 차이만 적용한다.

## v1 중단과 수정 근거

v1 본 평가 초반 여러 사례가 HTTP 400으로 종료했다. 실패 요청 1개를 재전송해 `max_tokens_exceeded`를 확인했다. 당시 연결부는 인자 후보를 얻기 위한 무료 list_actions 전체 응답을 매번 모델 히스토리에 추가했다. 모델이 요청하지 않은 대규모 행동 목록이 중복 삽입된 구현 문제다.

v1 원시 요청·결과와 실행 소스는 보존하며, 이 실행은 중단/입력 제약 확인 실험으로 보고한다. v2는 이미 일부 사례의 실행 오류를 관측한 후의 연결 수정이므로 완전한 최초 미관측 평가라고 주장하지 않는다. 진단 정답률에 따라 정책을 선택하거나 최적화하지 않았다.

## v2 차이

- 자동 list_actions 조회는 해당 도구의 인자 선택지 생성에만 사용한다. 전체 조회 결과는 helpers.jsonl에 보존하고 모델 히스토리에 자동 삽입하지 않는다.
- Jev가 직접 list_actions를 선택한 결과 및 모든 실제 선택 도구의 응답은 전체 히스토리에 그대로 남긴다. 전문가 요약·관측 절단·정답 기반 후보 필터는 없다.
- HTTP 오류의 안전한 error_type 식별자를 기록한다. 입력 한도 초과는 여전히 오류로 구분하며 자동 절단/재시도하지 않는다.
- 평가 원본 스냅샷·환경·채점·질문 순서·모델·히스토리 유지 정책은 변경하지 않는다.
- 개발 사례의 최대 3 turn smoke가 HTTP 200 요청 5회로 완료됐다. 종료 status turn_limit는 개발 smoke의 3 turn 제한이며 진단 성능 결과가 아니다.
- 결과는 별도 `results/netheal_official_jev_20261011_v2/`에 저장한다. 원본은 v1/upstream을 읽기 전용 참조한다.

## 실행

```powershell
.runtime/jev-official-venv/Scripts/python.exe -X utf8 netheal-topology-generalization/scripts/jev_official_eval_v2.py evaluate
```

100개 전체 결과는 v2 고정 설정으로 집계한다. 원본 공식 가중 집계 외에 전체 100개 분모의 성공률/미완료/API 오류를 보고한다. v1과 v2를 합쳐 서로 독립인 200개 사례라고 세지 않는다. 비용은 모든 시도·smoke·오류 진단 호출을 포함해 산출한다.
