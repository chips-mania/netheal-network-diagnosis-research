# LAYA 공식 환경 연결 전 전체 학습 자원 검증

작성일: 2026-10-12 KST. 실행 전에 기록한 자원 점검 명세다. 공식 100개 평가의 점수로 학습 설정을 선택하지 않는다.

## 순서

Jev v4 공식 100개 평가가 완료되었으므로 LAYA 후속을 시작한다. 기존 원본 환경·도구·채점 소스와 공식 스냅샷을 재사용한다. 먼저 사용자가 승인한 전체 Encoder+Head의 8비트 optimizer 실행 가능성을 검사한다. 기존 환경은 수정하지 않고 `.runtime/laya-official-venv`에 추가 패키지를 설치한다. GPU 작업은 순차 실행하고 FP32 가중치/연산을 유지한다. 유료 API는 호출하지 않는다.

## 고정 자원 점검 조건

- 기본 checkpoint: convaiinnovations/laya revision 7b928d828b7b0e022f929d9bd2e44165aa270148.
- bitsandbytes AdamW8bit, Encoder LR 2.5e-5, Head LR 1e-4, weight_decay 0.01, grad_clip 1.0.
- 전체 파라미터 requires_grad=True, Encoder와 Head gradient checkpointing, AMP 사용 안 함.
- microbatch 1, accumulation 1, seed 1012, 실제 optimizer update 8회.
- 자원 점검 입력은 과거 공개 관측 실험의 학습 prose 사례 4개(유형/위치 총 8질문). 공식 평가 사례를 쓰지 않는다. 최대 입력 1024, head 512. 잘림·선택지 손실은 허용하지 않는다.
- 모든 Encoder 블록 및 Head에 유한하고 0이 아닌 gradient와 실제 가중치 변화가 있는지 검사한다. 전체 loss·gradient, update 시간, PyTorch GPU peak allocated/reserved, nvidia-smi 장치 메모리·사용률, RAM을 기록한다.
- 8회 완료 후 전체 가중치와 optimizer를 저장하고 새 모델/optimizer에 로드한다. 동일 입력 logits 오차 1e-4 이하, 파라미터와 optimizer 저장 상태 일치 여부를 검사한다.
- OOM 또는 지원 오류는 실패로 기록하며 전체 미세조정 완료로 부르지 않는다. 8비트로 부족하면 LoRA+Head 조건을 별도 명세로 확정해 점검한다. 무단 FP16 전환은 하지 않는다.

## 판단과 후속

통과는 이 입력 길이에서 전체 backward/update/save/restore가 가능하다는 뜻이다. 본 학습 완료나 실제 진단 성능 개선을 뜻하지 않는다. 본 학습 데이터·도구 선택 연결·입력 길이·학습 예산은 자원 결과에 맞춰 별도 명세로 고정한 뒤 실행한다. 공식 평가 전에 학습 및 checkpoint 선택을 완료한다. 원본 Claude와의 Windows 실행·선택형 출력·문맥 처리 차이를 공개한다.

결과: `netheal-topology-generalization/results/laya_official_resource_20261012_v1/`. 기존 결과와 소스 스냅샷은 보존한다.

참고: [bitsandbytes 공식 설치 문서](https://huggingface.co/docs/bitsandbytes/installation), [LAYA 공식 학습 문서](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md). 2026-10-12 확인. 실제 호환 여부는 설치 성공이 아니라 GPU update로 확인한다.
