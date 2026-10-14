# 새 세션 인계 — 2026-10-12 KST

> **이 문서는 초기 상태의 보관 기록입니다. 다음 세션은 반드시 [최신 공통 입출력·학습량 비교 인계](2026-10-12_laya_common_io_handoff.md)를 먼저 읽으세요.** 아래의 기존 prepare 재개, 2,048회 단일 학습, 임의 조각에 전체 정답 부착, Jev 재호출 불필요라는 지침은 최신 논의로 대체됐습니다. 그대로 실행하지 마세요.

사용자 요청에 따라 LAYA 본 작업은 새 세션에서 이어간다. 이 인계 시점에는 새 학습을 실행하지 않았다. 프로세스 조회에서 `laya_official` Python 작업은 없었고 `prepared/`는 비어 있었다.

## Jev 완료

- 최종 보고서: `docs/results/netheal_official_jev_context_fix_20261012.md`.
- 원시 결과: `netheal-topology-generalization/results/netheal_official_jev_20261011_v4/`.
- `verification.json`: completed_verified, 100개 완료, 성공 42 / 오진 58 / 문맥 오류 0. 유형 정확도 71%, 위치 정확도 42%, 공식 가중 성공률 35.59%, 유형 macro-F1 0.6312.
- 원본 Claude: 성공 66/100, 공식 가중 성공률 65.04%, 유형 macro-F1 0.6373. 원본 환경·도구·채점 및 동일 100개 시나리오를 사용했지만 Windows 직접 handler 실행과 선택형 인터페이스라는 차이가 있으므로 완전히 동일한 실행 시스템은 아니다.
- 2,916개 요청 복원 검증, 100개 fingerprint, 소스 해시 검증 완료. 보고서에 재현 명령·비용·한계·중단 기록 포함. Jev 재호출 불필요.

## LAYA 현재 상태

완료:
- 새 환경 `.runtime/laya-official-venv`에만 bitsandbytes 0.50.2 설치. 기존 환경 보존. 실행 Python은 `.runtime/laya-official-venv/Scripts/python.exe -X utf8`.
- 전체 Encoder+Head FP32 AdamW8bit 자원 시험 두 건, 각각 **8 optimizer updates**. 본 학습 완료가 아니다. 저장 가중치와 optimizer 재로드 검증 완료.
- 짧은 입력(811–825 토큰): peak allocated 4,623,254,016 bytes, reserved 5,073,010,688 bytes.
- 긴 입력(3,097–3,123 토큰): peak allocated 5,108,872,704 bytes, reserved 7,495,221,248 bytes. Windows 공유 메모리 가능성이 있어 물리 6GB 내 실행이라고 주장하면 안 된다.
- 시험 산출물: `results/laya_official_resource_20261012_v1/`, `results/laya_official_resource_long_20261012_v1/` (모두 netheal-topology-generalization 아래).
- 공식 환경으로 train 40 / dev 20 생성 및 중복 감사 완료. 본 결과 폴더 `netheal-topology-generalization/results/laya_official_20261012_v1/`의 cases, snapshots, data_audit.json 확인.
- 학습 seed 94010000–94010039, 개발 seed 94020000–94020019. 공식 평가 사례와 seed·시나리오 중복 없음. 전문가 성공 여부로 선별하지 않음. 개별 관측 중복 수치는 감사 파일을 읽어 보고할 것.
- 원본 hierarchical 생성기의 요청 장비 수와 실제 장비 수가 다를 수 있다. 입력은 공개 응답의 실제 수 사용. 생성 당시 data_source.py와 현재 data.py의 state() 차이는 이 수정이며 입력 실행 명세에 기록했다.

중단/미완료:
- `scripts/laya_official_prepare.py` CPU 입력 준비는 시작했으나 완료 파일이 없다. 마지막 세션 출력은 unique 1,124개 중 25개 준비였다. 인계 시 실행 프로세스 없음, prepared 디렉터리 비어 있음. 준비 완료로 쓰지 말 것.
- capped probe, 2,048 updates 본 학습, 개발 평가, 공식 100개 LAYA 도구 평가는 **미실행**.
- LoRA 미실행. 메모리 제한 시험 실패 시 별도 조건 기록 후 적용 가능(사용자 승인됨).

## 다음 세션 실행 순서

1. 다음 명세와 코드를 읽고 실제 파일/프로세스를 다시 확인한다.
   - `docs/specs/laya_official_training_20261012.md`
   - `docs/specs/laya_official_input_execution_20261012.md`
   - `docs/specs/laya_official_memory_cap_20261012.md`
   - `netheal-topology-generalization/scripts/laya_official_data.py`
   - `netheal-topology-generalization/scripts/laya_official_input.py`
   - `netheal-topology-generalization/scripts/laya_official_prepare.py`
   - `netheal-topology-generalization/scripts/laya_official_capped_probe.py`
2. prepare.py는 prepared 디렉터리를 exist_ok=False로 생성한다. 기존 빈 디렉터리를 확인해 보존 이름으로 변경한 후 재실행하거나 안전한 재개 처리를 구현한다. 비어 있지 않으면 무작정 삭제하지 않는다. 전체 준비 결과는 마지막에 저장하므로 중간 중단은 재계산이 필요하다.
3. 준비 완료 후 가장 긴 실제 입력 3개로 allocator fraction 0.88의 capped probe를 실행한다. GPU 한 작업만 실행. 실패 시 기록 보존 후 LoRA 조건 사전 기록.
4. 본 학습 루프는 아직 구현되지 않았다. 새 base에서 전체 Encoder+Head FP32 AdamW8bit, encoder LR 2.5e-5 / head 1e-4, weight decay .01, clip 1, gradient checkpointing, seed 1012, microbatch 1, accumulation 1, cosine 2,048 updates/min LR 1e-6. tool/argument/fault/location 각각 512 updates. 1,024/2,048 checkpoint 및 정확한 RNG/optimizer/scheduler 재개 상태 저장. 이전 자원 시험 가중치를 출발점으로 쓰지 않는다.
5. 모든 공개 입력을 손실 없이 나누지만 학습은 질문당 조각 하나를 임의 선택하는 약한 라벨 학습이다. 추론은 모든 조각 확률 가중 평균. 정보 복원이 가능하다는 것과 모델의 통합 추론 능력을 구분한다. max_len 4096/head 1024, 잘림 금지. 소스와 조건을 학습 전에 동결한다.
6. 최종 checkpoint 동결 후 base 대비 train/dev 유형·위치 공동 정확도 등을 평가한다. 위치는 예측 유형으로 분기한다. 선택지 정순/역순/회전 비교. **dev 공동 정확도 +5%p 이상이고 train 공동 정확도도 개선**한 경우에만 공식 100개 전체 도구 평가 진행. 미달이면 판단 단계 결과와 미실행 범위를 보고한다. 개발 결과에 맞춰 반복 설정 변경하지 않는다.
7. 공식 평가가 허용되는 경우 기존 원본 100 snapshots와 도구·채점 코드, 100 환경 단계/100 solver turn/1200초 사용. GPU concurrency 1의 차이 명시. 평가에서는 미리 증거 수집하거나 정답으로 후보를 고르지 않는다. 유료 API 호출 없음.

## 주의

- 프로젝트 루트 작업, Windows PowerShell. GPU/Python 실행은 권한 상승이 필요할 수 있다.
- 디스크 잔여 마지막 확인 약 7.7GB. 전체 model+optimizer checkpoint는 건당 약 2.5GB. 실행 전 여유 확인하고 기존 산출물 삭제 금지.
- 기존 `src/run_nonlink_single_fault_audit.py` 변경을 건드리지 않는다. `.env` 출력 금지.
- 기존 Jev v4 실행/codec/tree partition 코드는 동결 상태이므로 수정하지 않는다. 재사용 시 import만 한다.
- 이번 공식 LAYA 본 학습/평가는 과거의 합성 과제 및 관측 표/문장 대조 실험과 별개의 새 실험이다. 실행 완료·중단·미실행을 분명히 기록한다.
