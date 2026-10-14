# LAYA 긴 입력 자원 점검 — 2026-10-12 KST

짧은 입력 811–825토큰의 전체 Encoder+Head 8비트 AdamW 8 updates 및 가중치/optimizer 저장·복원이 통과했다. 실제 사용된 PyTorch 최대 할당은 4,623,254,016바이트였다. 이는 긴 공식 도구 관측에서도 학습 가능함을 보장하지 않는다.

본 학습 조건을 고정하기 전에 같은 4개 기존 학습 사례의 공개 관측 문자열을 구분자와 함께 4회 반복하여 약 3,200토큰의 메모리 부하를 만든다. 새로운 증거/새 진단 사례라고 해석하지 않으며 반복된 정답 정보를 입력에 넣지 않는다. max_len=4096, head_max_len=1024, 그 외 8 updates/FP32/전체 Encoder/8비트 AdamW/gradient checkpointing/seed1012/저장·복원 조건은 짧은 점검과 같다.

긴 입력에서 OOM이면 그 상태를 보존한 뒤 LoRA+Head를 별도 사전 명세로 확정해 점검한다. 짧은 입력에서 이미 통과한 사실을 취소하거나 전체 Encoder 학습이 6GB에서 항상 불가능하다고 쓰지 않는다.

출력: `results/laya_official_resource_long_20261012_v1/`. 공식 100개 평가 사례와 유료 API는 사용하지 않는다.
