# official_benchmark/jev

2026-10-13 구조화 상태 실험은 `structured-validate`, `structured-evaluate` 진입점을 사용한다.
100개 평가 완료: JEV 48/100, 공개 Claude 66/100, 입력/API 오류 0건. `structured-audit`로 실제 입력 1,946건의 원본 로그 기반 검증을 재실행할 수 있다.
원본 도구 로그를 보존하고, 관측된 측정값·변화·호출 횟수·시각을 결정론적으로 누적한다.
입력 분할, 판단 점수 가중평균, 장애 원인 추론, 반복 행동 금지는 적용하지 않는다.
도구와 인자는 순차 선택하며 각 질문은 하나의 전체 구조화 상태를 받는다.
이 방식은 decision-oriented 압축이며 원본 전체를 복원하는 lossless codec이 아니다.
비교 대상은 기존 공개 Claude 100개 결과로, Claude의 입력 형식과는 차이가 있다.
실행 결과는 `runs/official_benchmark/jev/2026-10-13_structured_state_r01/`에 별도 보존한다.

Jev v4 100개 완료: 성공 42 / 오진 58 / 문맥 오류 0. 재실행 시 유료 API 가능.

`config.json`은 실행 진입점 목록이다. 기존 실험의 학습 조건을 새로 정의하거나 덮어쓰지 않는다. 실제 조건은 기존 명세와 동결 산출물을 따른다.

프로젝트 루트에서 `.runtime/venv/Scripts/python.exe -B -X utf8 experiments/official_benchmark/jev/run.py --list`로 확인한다. `run.py 단계`는 명령을 출력만 하고, `run.py --execute 단계 -- 기존인자`일 때만 실행한다.

기존 스크립트는 고정 출력 경로를 사용할 수 있다. 새로운 실험은 기존 결과를 덮어쓰지 않는 출력 설정을 먼저 마련해야 한다. LAYA GPU 작업은 한 번에 하나만 실행한다.
