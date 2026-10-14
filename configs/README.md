# 설정

기존 [config](../netheal-topology-generalization/config/)와 각 실행의 frozen protocol은 재현을 위해 유지한다. 실험별 새 진입점은 `experiments/<실험>/config.json`에 있다. 진입점 설정은 모델 학습 조건을 대체하지 않는다.

여러 새 실험에서 공유하는 설정만 이 폴더에 둔다. 비밀 API 키와 `.env` 내용은 포함하지 않는다. 현재 공통 설정 파일을 중복 복사하지 않았다.
