# 실험 진입점

- [일반화 v2](generalization/reliability_v2/)
- [가설 검증 v3](generalization/factorial_v3/)
- [Sol·Jev API pilot](api_pilot/)
- LAYA 대조: [학습](laya_controls/learning/) · [기호](laya_controls/symbols/) · [예산·순서](laya_controls/optimization/) · [관측 표현](laya_controls/observations/)
- 공식 환경: [Jev](official_benchmark/jev/) · [LAYA](official_benchmark/laya/)

각 `run.py`는 기본적으로 목록 또는 명령만 출력한다. 실제 실행은 `--execute`를 명시해야 한다. 기존 코드와 실험 조건은 보존되어 있으며 재구성 자체가 재학습이나 재평가를 의미하지 않는다.

코드 수정·새 실행 시 [구조 및 호환성 안내](../docs/architecture/README.md)를 먼저 확인한다.
