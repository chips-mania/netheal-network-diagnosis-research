# E3 — Exploration Collapse 진단 + 단일 개입 스펙 (v2)

## 0. 배경

Control / E1 / E2 모두 deterministic eval **AvgSteps = 2.0**. 보상을 여러 번 바꿔도 궤적 길이가 안 움직였다.

Reward reshaping은 **방문한 state** 위에서만 동작한다. 지금 가설은 policy가 `scan → diagnose`(step 2)로 굳어 도구 state를 샘플하지 않는다는 것이다.

E3는 이 가설을 **먼저 진단(Stage A)** 하고, **ent_coef만** 바꿔 확인한다. reward / action / E1 / E2와 섞지 않는다.  
`ent_coef=0.05` 보조 스캔은 하지 않는다. DSR 상승은 이번 성공 조건이 아니다.

**이미 확인됨:** 학습 코드는 `ent_coef`를 넣지 않았다. SB3 PPO 기본값은 **0.0**. E1 Control stdout은 2k에서 `ep_len_mean≈3.26`, `entropy_loss≈−3.42`, 30k에서 `≈2.64` / `−2.76`. 길이 collapse가 entropy 바닥보다 먼저다. `entropy_loss = −mean(H)`이므로 −3.4는 H≈3.4 nats(유효 ~30 action)이고, `log(72)≈4.3`과 맞으면 **남은 entropy는 도구가 아니라 72-way 진단**일 수 있다. Stage A는 이 해석을 `P(tool | post-scan)`으로 확인한다.

---

## 1. Stage A — 진단 (필수: action 분포)

entropy_loss만으로는 부족하다. 체크포인트마다 **post-scan 행동 분포**를 본다.

### 1.1 지표 (stochastic policy)

pre-scan valid는 `scan_network`뿐이라 첫 행동은 scan이다. 숏컷은 **step 2 diagnose**이지 step 0/1 diagnose가 아니다.

| 지표 | 의미 |
|---|---|
| `P(scan)` | 거의 1 (mask) |
| **`P(diagnostic \| post-scan)`** | scan 이후 스텝 중 ping/status/interfaces/traceroute 비율 |
| **`P(diagnosis \| post-scan)`** | scan 이후 스텝 중 최종 진단 비율 |
| `ep_len_mean` / entropy_loss | 학습 곡선 |

### 1.2 절차

- tensorboard는 기존 run에 없을 수 있다. **Control을 seed 1, 원본 reward, `ent_coef=0.0`으로 재학습**하고 logging callback만 추가한다.
- checkpoint: 2k, 5k, 10k, 20k, 30k (PPO iter에 맞춤). 각 시점에 stochastic rollout으로 위 확률을 계산한다.
- 재현성: 최종 deterministic DSR/AvgSteps가 이전 Control과 같은 패턴(숏컷, AvgSteps=2)이면 sanity OK.

### 1.3 판정

| 관찰 | 해석 |
|---|---|
| entropy가 5k 전 바닥 **그리고** `P(tool\|post-scan)`도 같이 죽음 | 이른 collapse. ent_coef가 먹힐 여지 |
| entropy는 중간(~3 nats)인데 **`P(tool\|post-scan)≈0`**, `P(diagnosis\|post-scan)`만 큼 | entropy가 진단 복권에 쓰임. **ent_coef=0.01은 진단을 더 흔들 뿐 tool-use를 안 열 가능성** |
| entropy 하락이 느리고 tool 확률도 서서히 줄음 | reward와 섞인 collapse. ent_coef만으로 부족할 수 있음 |

1차 개입값은 **미리 `ent_coef=0.01`로 고정**한다. 진단이 “값을 바꿔라”가 아니라 “0.01이 안 통할 이유를 주는지”만 본다. 사후 그리드서치 금지.

---

## 2. Stage B — E3 = Control + `ent_coef=0.01`

| 구성 요소 | Control | E3 |
|---|---|---|
| Terminal reward / 72-way | 원본 | 원본 |
| Gate / Factorized / Symmetric / E1 / E2 | 없음 | 없음 |
| `ent_coef` | **0.0** (SB3 기본, 코드에서 미설정) | **0.01** |

묻는 것: **exploration만 열어도 tool-use 궤적이 생기는가.** DSR이 아닌가.

---

## 3. 판정 (exploration 지표, DSR 아님)

Go/No-Go는 **stochastic eval**로 한다. deterministic eval은 숏컷을 강제로 재현하므로 AvgSteps가 항상 2.0이 된다. deterministic DSR은 참고열만.

사전 고정 (seed 1, E3 vs Control, stochastic):

```
No-Go:  AvgSteps < 3.0  AND  non-link tool-use-before-dx < 10%
Go:     AvgSteps ≥ 3.0  OR   non-link tool-use-before-dx ≥ 10%
```

- Go는 성공이 아니라 seed 2/3를 연다는 뜻이다.
- DSR이 떨어져도 위가 열리면 실패가 아니다 (탐색 trade-off).
- No-Go면 ent_coef를 올리지 않는다. 다음은 **초반 K step diagnose mask curriculum** (별도 스펙). 0.05 스캔 금지.

---

## 4. Stage C (Go일 때만, 지금 스펙 밖)

탐색이 열린 뒤에야 E1 shaping을 다시 얹을 수 있다. 지금은 손대지 않는다.

---

## 5. 체크리스트

- [x] Control `ent_coef` 기본 0.0 확인 (SB3 PPO, 우리 학습 코드 미설정)
- [x] Stage A: Control 재학습 + `P(tool\|post-scan)` / `P(diagnosis\|post-scan)` / entropy
- [x] E3: `ent_coef=0.01`만 변경
- [x] seed 1: 사전 규칙 **Go** (E3 sto tool-use 43%). 단 Control sto도 이미 35% — 탐색 부재가 아님
- [ ] seed 2/3: 규칙은 열리지만 Δ가 작음. 필요하면 별도 실행
- [x] DSR은 참고, 성공 조건 아님
- [x] `ent_coef=0.05` 돌리지 않음
