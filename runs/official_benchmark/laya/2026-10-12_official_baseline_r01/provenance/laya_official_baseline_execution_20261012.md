# LAYA 공식 100개 기본 모델 실행 명세

2026-10-12 KST. 사용자 승인: 공식 100개에서 LAYA 베이스모델 평가. Jev/Claude 재호출 없음.

- 기본 snapshot 7b928d828b7b0e022f929d9bd2e44165aa270148, 추가 학습 0회, FP32, seed 1012, 단일 GPU, allocator fraction .88.
- 개발 긴 입력에서 PyTorch native TransformerEncoder fastpath가 3.24GiB attention 할당으로 OOM을 일으켰다. 본 평가 전 MHA fastpath를 비활성화하여 SDPA 경로로 검증한다. 입력·가중치·정밀도는 유지하고 실패한 개발 기록을 보존한다.
- 동결 Jev v4의 episode 함수를 그대로 호출하며 프로세스 내 모델 client 생성자만 LAYA로 대체한다. 파일은 변경하지 않는다. 원본 환경/도구/후보 순서/분기/종료/채점 유지. 공개 tool_schemas도 같은 파일을 읽는다.
- 100개 초기 snapshot/fingerprint 검증. 환경 100단계, solver 100 turn, 에피소드 1200초. 준비 및 forward까지 시간에 포함하고 만료 후 도구 실행을 하지 않는다. 한 번의 동기 forward는 중간 취소가 불가능하므로 초과 시간이 약간 생길 수 있다.
- max_len/head_max_len=8192/8192. encoder 지원 상한 내에서 원본 선택지 보존을 위해 설정을 명시적으로 확장한다. 설치 LAYA의 48-token 개별 선택지 제한을 넘거나 head/instruction/state가 잘리면 오류로 기록한다. 모델이 해당 길이에서 좋은 성능을 낸다는 뜻은 아니다.
- Jev v4의 24000-byte 가드·분할·envelope·가중치를 먼저 그대로 적용한다. LAYA 토큰 한도를 넘을 때만 전체 원래 state를 동일 JSON-path envelope로 재분할하고 복원 검증한다. 선택지 전체 유지, tree weight=value JSON UTF-8 bytes. history weight=사건 수. 동률은 첫 후보. LAYA 단일 조각 choice=확률 argmax.
- 기본 checkpoint의 온도/선택지 수별 보정을 설치 runtime과 같은 clamp로 적용한다. 새 calibration은 하지 않는다. confidence를 확률 최댓값으로 대체하여 생성하지 않는다.
- 별도 개발 snapshot 3-turn smoke 및 development 원시 데이터 중 최대 trace에서 도구/인자 선택 긴 입력 추론을 검사한다. 본 평가 점수로 조건을 조정하지 않는다. 학습 자원 시험은 하지 않는다.
- 본 평가 전 소스·모델·토크나이저·snapshot 해시와 설정 동결. 완료 사례 재사용. 중단 시도 보존 후 해당 사례 처음부터 재개. 입력 오류/OOM/시간 초과/오답 자동 재시도 없음. 오류 사례도 전체 분모에 포함.
- 모든 입력·확률·선택·도구·결과 저장. 환경 단계·판단 질문·forward 수 구분. 원본 evaluator로 집계하고 Jev/Claude의 동일 사례와 비교한다.
- Windows 직접 handler, 로컬 GPU 동시성 1, 선택형 도구 인터페이스 및 LAYA 추가 분할 차이를 보고한다. 원본 Claude는 그대로 기준점으로 사용한다.
