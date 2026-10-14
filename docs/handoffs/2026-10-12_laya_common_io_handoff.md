# LAYA 새 세션 인계 — 공통 입출력과 학습량 비교

> **2026-10-12 사용자 정정:** 먼저 Jev·Claude와 동일한 공식 100개 스냅샷에서 기본 LAYA의 전체 에이전트 베이스라인을 평가한다. 기존 train 40/dev 20 고정 관측 평가는 이번 베이스라인이 아니다. 실행 우선순위는 [수정 계획](../specs/laya_official_baseline_plan_20261012.md)을 따른다. 아래 내용의 '진단 개선 후 전체 에이전트 평가' 순서는 이번 사용자 정정으로 변경됐으며, 공통 입력·비누출·자료 보존 원칙은 계속 적용한다.

## 공식 baseline 실행 중 — 2026-10-12 추가 기록

- 사용자 승인으로 `experiments/official_benchmark/laya/official_baseline.py evaluate` 실행 중. 출력: `runs/official_benchmark/laya/2026-10-12_official_baseline_r01/`.
- 원본 100개 fingerprint 검증, 개발 3-turn/긴 입력 smoke, 어댑터 테스트 3개 통과. FP32 8192-token 추론에서 PyTorch MHA fastpath OOM을 개발 단계에서 확인했고 비활성화 후 통과했다. 가중치·설치 패키지 변경 없음.
- `frozen_protocol.json`에 소스/모델/토크나이저/스냅샷 해시 동결. 기존 Jev episode 함수를 그대로 사용하고 모델 client만 프로세스 내 교체한다. [실행 명세](../specs/laya_official_baseline_execution_20261012.md).
- 기록 시점 첫 사례 1/100 완료, 다음 사례 진행 중. 최종 상태는 `manifest.json`과 실제 프로세스를 확인한다. 첫 사례의 완료는 정답을 뜻하지 않는다.
- 최초 숨김 supervisor PID 23596은 본 평가 시작 전 종료됐다. 중복 실행 자동 검토 거절 후 Win32_Process/GPU/로그로 실행 없음 확인, 직접 Python으로 시작했다. `supervisor.pid`는 실패한 시도의 역사적 기록이다.
- 현재 평가 PID와 감시 PID는 `watcher.json` 참조. `watch_official_baseline.py`는 CPU 감시·자원 기록 및 완료 후 보고서 실행만 한다. venv launcher와 실제 interpreter가 두 PID로 보일 수 있으며 GPU 평가 작업 두 개라는 뜻은 아니다.
- **기존 프로세스 확인 없이 다시 evaluate를 실행하지 않는다.** 완료 사례는 보존하고, 실제 중단이 확인되면 같은 명령으로 재개한다. 미완료 시도는 보존 후 처음부터 실행한다.
- 최종 보고서 예정 경로: `docs/research/official_benchmark/reports/laya_official_baseline_20261012.md`. 미완료 상태에서 결과가 생성됐다고 쓰지 않는다. 학습 및 Jev/Claude API 호출 없음.

아래는 본 실행 전 인계 내용이다. 학습·고정 관측 자료에 대한 과거 상태와 이번 공식 에이전트 실행 상태를 구분한다.

## 사용자가 원하는 비교

**동일한 관측·질문·선택지로 Jev / 기본 LAYA / 학습 후 LAYA를 비교한다.** 먼저 같은 데이터를 더 학습하면 성능이 개선되는지 확인한다. 새 학습 사례를 늘리는 효과는 반복 학습량의 효과와 분리한다.

기존 `laya_official_prepare.py`와 2,048회 단일 학습을 그대로 재개하지 않는다. 공통 입출력, 데이터 정답 타당성, 학습량별 비교 조건을 먼저 확정한다. 기존 동결 명세·소스는 역사적 기록으로 보존하고 새 구현·명세·실험 디렉터리를 사용한다.

## 실제 입출력 계약

Jev v4 요청은 다음 구조다. state 내용은 설명용 자리표시자다.

```json
{
  "model": "jev-1.13.0",
  "state": "<공개 관측과 이력을 압축한 JSON 문자열>",
  "questions": {
    "decision": {
      "type": "choice",
      "instructions": "<공통 지시문과 질문>",
      "criteria": {"c000": "<선택지 0>", "c001": "<선택지 1>"}
    }
  }
}
```

- state: 과제·목표·공개 힌트·장비 수·도구 명세·현재 단계·이전 도구/인자/공개 응답 이력. 숨겨진 장애·정답·전문가 결론은 입력하지 않는다.
- 출력: Jev의 `answers.decision.choice`와 `probabilities`. confidence는 별도 필드이며 확률 최댓값과 같다고 가정하지 않는다. LAYA도 같은 선택지 ID·확률 형식으로 변환한다. API model/id/usage나 내부 tokenizer까지 동일할 필요는 없다.
- 한 에피소드에서 여러 번 판단한다: 도구 선택 → 필요 시 인자 선택 → 환경 도구 실행 → 이력 갱신 → 반복. submit_diagnosis는 유형 선택 → **모델이 예측한 유형에 따른 위치 선택**으로 진행한다.
- 긴 입력을 분할하면 한 질문에 여러 호출이 필요하다. 환경 단계, 판단 질문, 조각 forward/API 요청 수를 구분한다. 기존 Jev 100개 평가에는 API 요청 2,916개가 있었다.
- LAYA는 도구를 직접 실행하거나 이전 호출을 자체 기억하지 않는다. 실행기가 도구를 호출하고 필요한 이력을 다음 입력에 넣는다. 이러한 각 판단을 지도학습 항목으로 만드는 것은 가능하지만 전체 도구 정책 학습 성공은 아직 검증하지 않았다.

## 현재 코드의 불일치 — 먼저 수정

| 항목 | 기존 Jev v4 | 현재 LAYA 준비 코드 |
|---|---|---|
| 일반 도구 인자 | 후보 255개 초과 시 source 먼저 선택 | 여러 source가 있으면 먼저 선택 |
| 링크 위치 | 링크 후보에서 직접 선택 | source 선택 후 링크 선택 |
| 입력 분할 | 요청 텍스트 24,000 bytes 가드 | max_len 4096/head 1024 토큰 검사 |

분할 envelope와 문구 차이도 있다. 새 공통 adapter는 같은 상태에서 같은 질문·후보 순서·입력 문자열·분할 경계·결합 가중치·선택 규칙을 생성해야 한다. 두 모델의 한도를 모두 통과하는 조각을 한 번 만들어 그대로 사용한다. 무손실 복원 가능성이 모델의 통합 추론 능력을 보장하지 않는다.

기존 Jev v4 소스를 수정하지 않는다. 공통 계약을 바꾸면 기존 Jev 42/100은 역사적 결과로 보존하고, 동일 조건 비교에는 새 계약에서 Jev 재측정이 필요하다. 과거 유료 호출 승인 이력은 있으나 이번 인계 갱신 요청만으로 새 재측정을 시작하지 않는다. 계약·실행 범위·비용을 먼저 구체화한다.

## 학습 데이터와 정답

개념적 저장 형식은 `{case_id, input: {state, questions}, target: {choice}, metadata}`다. LAYA에는 같은 input과 별도 expected를 전달한다. **target/expected는 loss·채점에만 사용하며 모델 입력에 넣지 않는다.** 학습과 추론의 입력 구조를 동일하게 한다.

1. 기존 train 40/dev 20의 원시 snapshots·공개 trace·truth는 재사용 후보다. 새 공통 규칙으로 문제를 재생성하고 다시 검증한다. 데이터 크기는 새 설계에서 확정한다.
2. 유형·위치 정답은 시뮬레이터에서 가져온다. 입력과 후보 생성은 공개 정보만 사용한다. 후보를 정답으로 좁히지 않는다.
3. 도구·인자에는 유일한 최적 정답이 자동으로 존재하지 않는다. 현재 scan→status 전체→interfaces 전체→공개 연결 ping→진단은 고정 수집 정책 모방이며 최적 전문가가 아니다. Jev의 답도 검증 없이 정답으로 쓰지 않는다.
4. 먼저 충분한 공개 관측에서 유형·위치를 판단하는 학습 효과를 검증하는 방향이다. 도구 정책 학습에는 여러 타당한 행동, 관측 부족·실수 이후 상태의 수집과 라벨링을 별도 설계한다. 정답 경로만 학습하면 실행 시 모델 행동으로 생긴 낯선 상태에 약할 수 있다.
5. **일부 입력 조각에 전체 사례 정답을 그대로 붙이는 이전 방식은 재사용하지 않는다.** 모든 조각의 출력을 결합한 상태 단위 loss 등 학습·추론 일치 방식을 검토하되 구체적 해법은 미확정이다. 메모리 점검 전에 타당성을 확인한다.
6. 충분한 관측의 정의를 정답이나 모델 성공률로 임의 선별하지 않는다. 고정 공개 수집 절차와 대조 검증으로 진단 가능성·모호함을 점검하고 불충분/제외 사례를 기록한다.
7. split은 **원본 사례 단위**다. 같은 사례의 여러 시점·질문·선택지 순서 변형·조각은 같은 split에 묶는다. seed뿐 아니라 snapshot/fingerprint와 동일·준동일 공개 관측을 검사한다. 성공 사례만 선별하지 않는다.

## 비교와 학습량 실험

- 고정 관측 비교: 동일한 문제를 Jev·기본 LAYA·학습된 LAYA에 준다. 이것은 전체 도구 탐색 성능과 구분한다.
- 동일 학습 데이터·loss·유효 배치·질문 비율·입력 형식을 유지하고 update 수만 변화시킨다. 예시 checkpoint **0/256/1024/2048/4096**은 논의용이며 **확정된 예산이 아니다**. seed 수도 사전에 정한다.
- 학습률 스케줄과 데이터 노출 순서도 통제한다. 예산마다 cosine 길이까지 바꾸면 학습량과 스케줄 효과가 섞인다. 동일 update라도 배치가 다르면 학습 질문 수가 달라진다.
- 각 사전 지정 checkpoint에서 train과 dev 곡선을 기록한다. 예정된 개발 평가와 결과를 보고 조건을 다시 바꾸는 튜닝을 구분한다. 최종 checkpoint 선택 규칙과 평가 횟수를 미리 정하고 최종 평가셋을 반복 튜닝에 사용하지 않는다.
- 순차 진단에서는 각 모델의 예측 유형이 다르면 위치 질문도 달라질 수 있다. 이는 **동일한 분기 규칙**을 적용한 결과다. 정답 유형을 주는 위치 평가는 보조 평가로 별도 표시한다.
- 전체 에이전트 비교에서는 같은 초기 사례·도구·제약을 사용하되, 모델 행동이 달라지면 이후 관측까지 동일할 수는 없다. 평가 때 증거를 대신 수집하지 않는다.
- train/dev 모두 상승하면 학습 부족 가설을 지지한다. train만 상승하면 일반화 한계, 둘 다 낮으면 입력·라벨·최적화 문제를 점검한다. 실패를 LAYA 능력 상한으로 단정하지 않는다.

기록 지표: 유형 정확도/macro-F1, 위치·공동 정확도, 선택지 순서 민감도, 실제 updates·처리 질문/토큰 수, 환경 단계·모델 호출 수, 시간·GPU/RAM, 저장/복원 검증. 판단 개선 전 전체 탐색을 무작정 재실행하지 않는다. 이전 dev +5%p 및 train 개선 게이트는 40/20 단일 학습 초안의 기준이며 새 설계의 게이트는 평가 전에 다시 확정한다.

## 공식 학습법과 자원 조건

[공식 안내](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md)와 설치된 `.runtime/laya-venv/Lib/site-packages/laya/train.py`를 대조했다. 새 실행 전 참조 버전과 소스 해시를 고정한다.

- 공식 기본: 전체 Encoder+Head, Encoder LR 2.5e-5/Head LR 1e-4, AdamW·cosine, RLCD. soft-CE와 선택지 섞기도 지원한다.
- 공식 예시 유효 배치는 64다. 6GB에서 microbatch 1을 사용해도 gradient accumulation으로 유효 배치를 늘릴 수 있다. 이전 accumulation 1은 필수 조건이 아니다.
- 공식 4 epochs를 다른 크기 데이터에 그대로 적용하지 말고 실제 updates와 질문 노출 수를 계산한다. 공식 예시가 충분한 학습량을 보장하지 않는다.
- GTX 16xx에서 FP16이 느리면 FP32를 사용한다는 안내가 있다. 사용자 승인대로 **FP32 전체 Encoder+Head AdamW8bit 먼저, 부족하면 LoRA**를 검토한다. GPU 작업은 하나씩 실행한다.
- 이전 40/20 사례·2048 updates·4096/1024 길이·조각별 약한 라벨 방식은 자체 설계다. 공식 레시피를 그대로 재현했다고 표현하지 않는다.
- 공식 학습의 별도 calibration 데이터와 확률 보정도 고려한다. 특히 조각 확률 결합이나 confidence 사용 시 보정 여부·분리·규칙을 명시한다. 두 모델의 보정 상태가 같다고 가정하지 않는다.

## 실제 완료 상태와 위치

- Jev v4: 100개 완료, 성공 42/오진 58/문맥 오류 0, 유형 정확도 71%, 공동 정확도 42%, 공식 가중 성공률 35.59%, macro-F1 0.6312. [최종 보고서](../research/official_benchmark/reports/netheal_official_jev_context_fix_20261012.md).
- Jev 원시 결과: `runs/official_benchmark/jev/netheal_official_jev_20261011_v4/`. 원본 공식 환경·스냅샷은 `netheal-topology-generalization/results/netheal_official_jev_20261011_v1/upstream/`에 유지.
- LAYA 전체 Encoder FP32 8-bit 자원 시험 두 건: 각각 **8 updates 및 저장/복원 검증 완료**. 본 학습 완료가 아니다. 짧은 입력 peak allocated/reserved 4,623,254,016/5,073,010,688 bytes. 긴 입력 5,108,872,704/7,495,221,248 bytes로 물리 6GB 내 실행을 보장하지 않는다.
- 자원 시험 산출물: `runs/official_benchmark/laya/resource_probes/`. 기본 checkpoint는 `.runtime/huggingface/hub/models--convaiinnovations--laya/snapshots/7b928d828b7b0e022f929d9bd2e44165aa270148`.
- train 40/dev 20 원시 데이터 생성 완료: `netheal-topology-generalization/results/laya_official_20261012_v1/`. 학습 seed 94010000–94010039, 개발 seed 94020000–94020019. 기존 감사에서 공식 사례와 seed/fingerprint 중복 없음. 개별 관측 중복은 data_audit.json 참조. 새 문제 생성 후 다시 검사한다.
- 공식 hierarchical 생성기는 요청한 장비 수와 실제 장비 수가 다를 수 있다. 입력은 공개 응답의 실제 수를 사용한다.
- 입력 준비는 중단됐고 완료 산출물 없음. 가장 최근 확인 당시 prepared/는 비어 있었고 실행 프로세스가 없었다. 다음 세션에서 다시 확인한다.
- **capped probe·본 학습·개발 평가·LAYA 공식 100개 평가·LoRA는 미실행.** 새 학습 루프도 미구현이다.

## 다음 세션 순서

1. 이 문서와 [구조 안내](../architecture/README.md), 실제 프로세스·파일·디스크 상태를 확인한다.
2. 기존 Jev v4 요청 로그와 `jev_official_eval_v4.py`, `laya_official_data.py`, `laya_official_input.py`를 대조하고 새 공통 adapter 계약·동등성 테스트를 만든다.
3. 사례 단위 split과 타당한 정답을 갖춘 공통 문제를 새 폴더에 생성한다. 기존 자료는 보존한다.
4. 데이터 수·seed·loss·유효 배치·LR 스케줄·학습량별 checkpoint·보정·평가 규칙을 새 명세에 실행 전에 고정한다.
5. 새 실제 입력과 누적 방식으로 메모리 상한 시험을 한다. 기존 제안은 가장 긴 입력 3개, allocator fraction 0.88이었다. 실패 시 기록 후 LoRA 조건을 별도로 고정한다.
6. 기본 checkpoint에서 학습 전 비교와 학습량 실험을 진행한다. 자원 시험 가중치를 이어 쓰지 않는다. optimizer/scheduler/RNG/update 수가 복원 가능한 checkpoint를 저장한다.
7. 판단 개선 및 도구 정책 준비 후에만 공식 전체 에이전트 비교를 진행한다. 원본 100 환경 단계/100 solver turn/1200초, Windows 직접 handler·단일 GPU 동시성 1의 차이를 명시한다. 이미 Jev 결과를 본 100개는 프로젝트 전체의 새 미관측 평가가 아니다.

## 운영 주의

- Windows PowerShell, GPU는 GTX 1660 SUPER 6GB. LAYA 환경 `.runtime/laya-official-venv/Scripts/python.exe -B -X utf8`, 공식 환경용 `.runtime/jev-official-venv/Scripts/python.exe`, 기존 시뮬레이터 `.runtime/venv/Scripts/python.exe`. 기존 환경을 덮어쓰지 않는다.
- Git 임시 pack 5개는 사용자 승인으로 11,188,207,541 bytes 삭제했다. 삭제 직후 C: 여유 **20,595,458,048 bytes**였다. model+optimizer 저장은 건당 약 2.5GB이므로 여러 checkpoint 보존량을 계산하고 실행 직전 여유를 다시 확인한다. 자원 시험 가중치는 삭제하지 않았다.
- 기존 `src/run_nonlink_single_fault_audit.py` 수정, 동결 Jev 코드, 과거 결과를 건드리지 않는다. `.env` 키 출력 금지.
- 기존 결과 경로의 junction/hard link를 별도 사본으로 취급하지 않는다. 새 소스·산출물은 새 경로를 사용한다.
- `experiments/official_benchmark/laya/run.py`는 이전 실행기의 진입점이다. 최신 공통 입출력 학습 구현이 아니므로 바로 --execute로 재개하지 않는다.
- 이번 수정은 인계 문서 갱신이며 새로운 학습·API 호출·전체 재평가의 실행 완료를 뜻하지 않는다.
