# NetHeal 토폴로지 일반화 실험

프로젝트의 [목적·결과·설치 안내](../README.md)와 [연구 문서 인덱스](../docs/README.md)를 참고하세요.

- `src/`: PPO·BC·GNN 정책, 환경 래퍼, 감사 및 실험 실행 모듈
- `config/`: 학습·평가 프로토콜
- `reports/`: 실험별 보고서
- `results/`: 구조화된 실험 기록
- `models/checkpoints/`: 현재 보존된 학습 가중치 (`_tmp_*`는 로컬 전용)

모듈은 이 디렉터리에서 `python -m src.<module>`로 실행하며, 형제 디렉터리 `netheal-ai-agent-benchmark`를 `PYTHONPATH`에 추가해야 합니다. `run_*` 실행 시 보고서·결과·모델이 갱신될 수 있습니다.
