# LAYA 공통 입력 baseline v1 — 실행 전 고정

- 범위: 기존 60개 원시 사례 전부(train 40 / development 20), 기본 checkpoint, updates=0. 새 학습·Jev API·공식 100개 에이전트 평가는 실행하지 않는다.
- 관측: 원시 고정 수집 trace에서 최초 submit_diagnosis 이전까지만 사용. 마지막 정답 제출과 응답, truth, teacher_result, questions는 입력에 넣지 않는다. 공개 catalog만으로 유형 및 각 유형의 전체 위치 후보를 만든다. 링크는 직접 선택한다.
- 자료의 관측 충분성은 아직 입증되지 않았다. 관측 부족/중복 여부를 기록하되 모델 성공이나 정답에 따라 사례를 제외하지 않는다. 이 결과는 고정 수집 절차의 baseline이며 충분한 증거에서의 능력 상한을 뜻하지 않는다.
- 공통 지시문/질문은 Jev v4와 동일. 모델에 보낼 input을 저장한다. Jev는 추후 같은 input에 model 필드만 더하면 된다. 새 tree 분할 계약이므로 과거 Jev 42/100과 직접 비교하지 않는다.
- LAYA FP32, 단일 GPU, microbatch 1, inference_mode, seed 1012. max_len=4096/head=1024, Jev 24000-byte 가드를 함께 통과하고 truncation/선택지 손실이 없는 조각만 사용한다. JSON-path 무손실 복원 검사, subtree JSON UTF-8 바이트 가중 평균, 동률은 첫 후보.
- 확률: raw logits softmax(T=1). 기존 checkpoint의 온도 보정은 적용하지 않으며 재보정도 하지 않는다. confidence는 생성하지 않는다. 학습 비교에서는 동일 확률 규칙을 유지해야 한다.
- 주 평가: 유형 예측 후 예측 유형의 위치 선택. 보조: 정답 유형 조건 위치 정확도(oracle-type; 주 평가와 명확히 구분). 두 방향 링크 표기를 같은 무방향 링크로 채점한다. 장비는 문자열 일치.
- 순서 민감도: 모든 사례에서 유형 후보 역순 및 canonical 예측 유형의 위치 후보 역순을 각각 평가, 의미상 선택 변경률 기록. 결과를 보고 순서를 채택하지 않는다.
- 지표: split별 유형 accuracy/macro-F1, 위치 accuracy, 공동 accuracy, oracle-type 위치 accuracy, 순서 변경률, 질문/forward/토큰 수, 시간, GPU allocated/reserved와 RAM.
- 학습 loss/스케줄/업데이트 예산은 이번 0-update 평가와 별개이며 아직 확정하지 않는다. 조각별 정답 라벨 학습을 만들지 않는다.
- 재현성: 소스/설치 LAYA/기본 모델/입력 사례 해시와 config를 실행 폴더에 저장, 질문별 결과 즉시 저장. 기존 결과와 가중치는 변경하지 않는다.
