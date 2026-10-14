# NetHeal 토폴로지 일반화 실험

2026-10-12부터 새 실험 진입점은 [experiments](../experiments/README.md), 결과의 실제 위치는 [runs](../runs/README.md), 재사용 코드는 [src/netheal_research](../src/netheal_research/README.md)에 분리했습니다. 이 폴더에는 기존 실행 코드와 호환 경로를 보존합니다. [구조 안내](../docs/architecture/README.md)를 참고하세요.

프로젝트의 [목적·결과·설치 안내](../README.md)와 [연구 문서 인덱스](../docs/README.md)를 참고하세요.

- `src/`: PPO·BC·GNN 정책, 환경 래퍼, 감사 및 실험 실행 모듈
- `scripts/`: [LAYA·Jev 실행, 자원 시험, 검증, 보고서 생성](scripts/README.md)
- `config/`: 학습·평가 프로토콜
- `reports/`: 실험별 보고서
- `results/`: 구조화된 실험 기록
- [결과 디렉터리 목록·용량·상태](results/README.md)
- `models/checkpoints/`: 현재 보존된 학습 가중치 (`_tmp_*`는 로컬 전용)

모듈은 이 디렉터리에서 `python -m src.<module>`로 실행하며, 형제 디렉터리 `netheal-ai-agent-benchmark`를 `PYTHONPATH`에 추가해야 합니다. `run_*` 실행 시 보고서·결과·모델이 갱신될 수 있습니다.
