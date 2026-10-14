# 공통 코드

- `paths.py`: 작업 공간 기준 경로 처리와 외부 경로 이탈 방지.
- `experiment_runner.py`: 실험별 설정에서 명령을 구성하고 명시적인 실행 요청이 있을 때만 기존 실행기 호출.
- `evaluation/metrics_utils.py`, `evaluation/research_metrics.py`: 기존 지표 구현을 내용 변경 없이 이전.
- `environments/controlled_faults.py`: 기존 장애 제어 구현을 내용 변경 없이 이전.

이전한 기존 모듈 3개는 옛 src 경로에도 hard link가 있어 기존 import가 유지된다. 다른 정책·학습 코드의 새 패키지 이식은 후속 전환 대상이며 동결 소스는 원위치에 보존했다. 모델별 의존성은 기존 전용 가상환경을 사용한다.
