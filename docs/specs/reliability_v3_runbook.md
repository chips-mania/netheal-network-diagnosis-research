# v3 실행 및 산출물 확인

LAYA/Jev는 이번 실행에서 제외했다. [가설·범위·판정 기준](reliability_v3_hypotheses.md)을 먼저 읽는다.
이번 실행은 순위 처리 × 학습 구조의 3×2 실험이며 정상/복합 장애·실장비 단계는 별도다.

## 실행

프로젝트 루트에서 Python 환경을 준비한 뒤:

```powershell
$env:PYTHONPATH = (Resolve-Path ./netheal-ai-agent-benchmark).Path
cd netheal-topology-generalization
python -m pytest tests/test_factorial_v3.py tests/test_research_suite.py -q

# 출력 폴더는 새 이름이어야 한다. 기본값이 사전 고정 본 실험이다.
python -u -m src.run_factorial_v3 --out results/factorial_v3_new
python -m src.report_factorial_v3 results/factorial_v3_new
```

실제 2026-10-08 실행은 저장소 루트의 .runtime/venv 환경을 사용했다.
report_factorial_v3는 가설 대비 계산과 무결성 검증도 함께 수행한다.
수치만 다시 계산하려면 다음을 실행한다.

```powershell
python -m src.analyze_factorial_v3 results/factorial_v3_new
```

## 중간 상태와 완료 상태 구분

- manifest.json의 status=training: 학습 중. 최종 진단 정확도를 보고하면 안 된다.
- training_complete_manifest.json: 모든 학습 체크포인트가 저장된 시점과 해시.
- manifest.json의 status=training_complete: 학습 완료, 최종 평가 진행 중일 수 있다.
- manifest.json의 status=completed: 예정된 모든 최종 평가가 끝났다.
- verification.json의 status=passed: 원자료와 요약 통계·공통 사례·해시까지 검증했다.

로그가 끊겨 보이더라도 실행 프로세스와 manifest를 확인해야 한다.
학습은 각 모델 시작 시 로그를 남기며 매 update를 출력하지 않는다.
이 실행기는 자동 중단 재개 기능을 제공하지 않는다. 중단되면 부분 산출물을 보존하고
새 폴더에서 재실행해야 하며, 부분 평가를 완전한 실행으로 보고해서는 안 된다.

## 보존 파일

- frozen_spec.md / protocol.json: 실행 전 설계와 실제 인자.
- manifest.json: 실행 환경, 소스 및 체크포인트 해시, 완료 상태.
- demonstrations_*.pt / selected_demos_*.json: 원본 데모와 군별 선택 목록.
- demo_audit_*.json: 실패를 포함한 전문가 데모 수집 시도.
- training.json / checkpoints/: 학습 비용과 모델.
- episodes_*.jsonl: 모든 에피소드의 정답·예측·도구 비용·단계·실행시간.
  상세 도구 trace는 각 조건 첫 4개 사례에 저장한다.
- network_probes.jsonl: 정책에 제공하지 않은 공통 all-pairs 네트워크 측정.
- summaries.json: 모델/반복/명명조건/토폴로지별 지표.
- hypothesis_results.json: H1/H2 주요 대비와 보조/탐색 대비.
- secondary_comparisons.json: 세부 조건별 탐색 비교.
- report.md / factorial_exact.png: 표와 그림.
- verification.json / analysis_manifest.json: 검증 결과 및 분석 코드 해시.

v2 파일과 실행 시점 소스는 변경하지 않는다. 새 분석 스크립트가 추가되는 것은 허용하지만
기존 manifest에 기록된 소스의 내용을 바꾸면 해시 검증이 실패한다.
최종 문서는 3개 replica의 변동성을 포함하며, 데이터 수집과 초기화의 변동을 별도로 분리한 것으로 해석하지 않는다.
