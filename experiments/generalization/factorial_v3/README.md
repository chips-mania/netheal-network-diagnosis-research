# generalization/factorial_v3

ID·순서와 토폴로지 학습 구조 3×2 실험. 기존 결과 완료.

`config.json`은 실행 진입점 목록이다. 기존 실험의 학습 조건을 새로 정의하거나 덮어쓰지 않는다. 실제 조건은 기존 명세와 동결 산출물을 따른다.

프로젝트 루트에서 `.runtime/venv/Scripts/python.exe -B -X utf8 experiments/generalization/factorial_v3/run.py --list`로 확인한다. `run.py 단계`는 명령을 출력만 하고, `run.py --execute 단계 -- 기존인자`일 때만 실행한다.

기존 스크립트는 고정 출력 경로를 사용할 수 있다. 새로운 실험은 기존 결과를 덮어쓰지 않는 출력 설정을 먼저 마련해야 한다. LAYA GPU 작업은 한 번에 하나만 실행한다.
