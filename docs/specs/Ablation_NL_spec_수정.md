# Ablation-NL — Non-Link-Only 학습 분포 실험 스펙

## 0. 배경 및 질문

**Control 가중치 (E3 Control checkpoint 하나로 고정, seed 1)**: 4-fault deterministic 참고값은 Exact DSR 20%, Link 56%, Perf 24%, Device 0%, Misconfig 0%. 검산: (56+24+0+0)/4 = 20. 같은 seed 1 재학습마다 Exact가 7%→10→20%로 흔들린 이력이 있다(PPO/env 샘플 노이즈 가능). **표의 Control 숫자는 그 25-ep 슬라이스를 재사용하지 않는다.** 같은 zip을 Ablation-NL과 **동일한 non-link-only eval 프로토콜**(episode 수, seed, sto+det)로 다시 평가한 값을 쓴다.

가설: **Link가 "scan 후 즉시 진단"만으로 완벽히 풀리는 값싼 subtask라서, 매 episode 학습 신호에 계속 경쟁 상대로 끼어들어 나머지 세 fault(더 어렵고, 정책 복잡도가 높은)에 대한 학습 신호를 구조적으로 억누른다.**

**9.9와의 관계 (prior 조정)**: 9.9에서 이미 fault 하나씩만 30k **timestep** 단독 학습한 결과가 있다 — Link 경쟁이 0이고, 해당 fault 노출량은 이번 Ablation-NL(30k timestep을 3-way로 나눠 대략 **~10k timestep/fault**)의 약 3배인데도 Device DSR 0%, Perf 19%, Misconfig 22%였고 Perf/Misconfig도 exact policy가 아니라 최빈 location 복권에 가까웠다. 즉 **"경쟁을 없애고 노출량을 늘려도 Device는 안 풀렸다"**는 훨씬 강한 조건의 결과가 이미 있다. 이건 "Link 제거가 non-link를 연다"는 가설의 prior를 낮춘다.

그래도 Ablation-NL은 9.9와 다른 질문을 묻는다: 9.9는 "specialist가 그 fault 하나를 푸는가"이고, Ablation-NL은 "**4-way 혼합에서 greedy shortcut이 Link 때문에 고정되는가**"이다. 후자는 아직 확인한 적 없다. 따라서 이번 실험에서 **"Device/Misconfig의 0%가 깨진다"는 기대는 낮게 잡고**, 더 현실적인 관찰 대상은 "Link를 없앴을 때 deterministic shortcut/AvgSteps 패턴 자체가 조금이라도 흔들리는가"이다.

E1(dense shaping)/E2(timeout 비대칭)/E3(exploration) 모두 이 구조를 그대로 둔 채 aggregate reward 안에서 뭔가를 조정했다. 만약 진짜 병목이 "Link와의 경쟁"이라면, aggregate 안에서 뭘 바꿔도 이 쏠림 자체는 안 없어진다. 이 실험은 **reward도 action space도 observability도 안 건드리고, 학습 시 fault 샘플링 분포에서 Link를 제외**해서 이 가설을 직접 확인한다. E1/E2/E3와 섞지 않는다.

---

## 1. Ablation-NL = Control + 정확히 한 가지

| 구성 요소 | Control | Ablation-NL |
|---|---|---|
| Algorithm | MaskablePPO | 동일 |
| Terminal reward | 원본 그대로 | **원본 그대로, 수정 금지** |
| 72-way diagnosis action space | 원본 (link 포함 4 type) | **동일, 축소하지 않음** — agent는 여전히 link를 답으로 고를 수 있음 (다만 학습 중 GT가 link인 적이 없으므로 항상 오답) |
| Gate / Factorized / Symmetric | 없음 | 없음 |
| E1 shaping / E2 timeout 비대칭 | 없음 | **없음** |
| **Fault 샘플링 분포 (학습 시 GT 생성)** | {link, device, perf, misconfig} 균등(가정 25%씩) | **{device, perf, misconfig} 균등(33.3%씩), link 완전 배제** |

**중요**: action space나 diagnosis 로직은 그대로 둔다. 학습용 episode generator에서 GT fault_type을 뽑는 샘플러 하나만 바꾼다. Link의 scan observability(missing edge가 바로 보이는 것)는 전혀 안 건드린다 — 이건 학습 분포 조정이지 환경 수정이 아니다. `NetworkTroubleshootingEnv(fault_types=…)`는 GT 샘플러만 바꾸고 diagnosis action은 `list(FaultType)`로 남아 72-way가 유지된다.

---

## 2. 알려진 confound (결과 해석 시 반드시 명시)

학습은 **30k timestep**이다 (30k episode가 아님). E3 Control 평균 길이가 약 2.6이면 전체 episode는 대략 1만 개 남짓. Control에서 non-link는 episode의 약 3/4이므로 fault당 대략 **~2.3k episode / ~7.5k timestep**. Ablation-NL은 전부 non-link라 fault당 대략 **~3.1k episode / ~10k timestep** — 노출량 약 **1.33배**.

즉 이 실험에서 non-link DSR이 오른다면, 그게 **(a) Link와의 경쟁이 사라져서**인지 **(b) 단순히 각 fault의 학습 데이터가 늘어나서**인지 이 ablation 하나로는 분리가 안 된다. 1차 검증으로는 충분하다. 원인을 더 분리하고 싶다면 후속으로 "Link 포함하되 빈도만 줄인 버전"(예: link 10%, 나머지 30%씩)을 별도로 돌려볼 수 있다 — 지금 단계에서는 안 한다.

---

## 3. 평가 (Eval)

### 3.0 Control 비교 기준 고정

E3 Control **zip을 재학습하지 않는다** (새 Control은 또 다른 DSR이 됨). 그 가중치를 Ablation-NL과 **같은 non-link-only 프로토콜**로 재평가한다. 4-fault 100-ep det는 zip identity 확인용 참고열만 (E3 공개값 Exact 20%).

zip이 없으면 그때만 4-fault Control을 동일 코드·seed로 재학습하고, 그 값을 새 기준으로 쓴다. 서로 다른 시점의 재학습값을 표에 섞지 않는다.

### 3.1 Primary eval — non-link-only 분포

학습 분포와 동일하게 {device, perf, misconfig}만으로 eval episode 생성 (100 episode, 순환 샘플 → fault당 ~33). Control과 Ablation-NL을 **같은 seed·같은 n**으로 돌린다.

### 3.2 Secondary eval — 원래 4-fault 혼합 분포 (참고용, 판정에 안 씀)

Ablation-NL 모델을 {link, device, perf, misconfig}에서 eval. Link에 어떻게 반응하는지만 본다.

### 3.3 Stochastic + Deterministic 둘 다 (E3와 동일 관례)

판정(§5)의 0% 벽·+5%p·shortcut/AvgSteps는 **deterministic(greedy)** 기준. stochastic은 표에 같이 기록.

### 3.4 Evidence Acquisition 정의 (E1 Φ와 혼동 방지)

E1 shaping은 쓰지 않는다. 측정만 한다 (`pbrs_apply_shaping=False`).

- **Evidence Acquisition (rich)**: E1 스펙 §2.4 predicate를 eval 전용으로 재사용. diagnose 이전에 type/loc bit가 켜졌는지만 측정.
- **Evidence-Consistent Diagnosis**: 확보한 bit가 예측 type/location과 맞는지 (E1 `evidence_consistency`).
- **Tool-use rate (simple, E3 호환)**: diagnose 이전 diagnostic tool ≥1.

---

## 4. 지표 표

### 4.1 Primary (seed 1, stochastic + deterministic)

| Metric | Control (NL eval, stochastic) | Control (NL eval, det) | Ablation-NL (stochastic) | Ablation-NL (det) |
|---|---|---|---|---|
| Non-link DSR | | | | |
| Device DSR (exact n / n) | | | | |
| Perf DSR | | | | |
| Misconfig DSR (exact n / n) | | | | |
| Tool-use rate (pre-diagnose) | | | | |
| Evidence Acquisition | | | | |
| Evidence-Consistent Diagnosis | | | | |
| Shortcut Rate | | | | |
| AvgSteps | | | | |

### 4.2 Secondary (참고, Ablation-NL 모델을 원래 4-fault 분포에 eval)

| Metric | 값 |
|---|---|
| Link DSR (Ablation-NL 모델) | |
| 전체 4-fault DSR (Ablation-NL 모델) | |

---

## 5. 판정 기준 (3단, Perf 단독 상승을 Go로 오인하지 않도록 분리)

판정은 **deterministic primary (NL eval)** 에 앵커링한다. n≈33에서 1 exact는 약 3%이므로 “0%를 벗어남”은 한 방으로 보지 않는다.

1. **가설 지지**: Device 또는 Misconfig에서 **exact ≥ 2 episode** 이고, 그 exact 중 Evidence-Consistent도 **≥ 2**. (우연 복권이 아니라 evidence 기반이라는 최소 근거)
2. **Go (seed 2/3 진행)**: 위 조건 + non-link DSR(det aggregate)이 Control NL eval 대비 **+5%p 이상**
3. **No-Go**: Device/Misconfig det exact < 2 (또는 consistent < 2), 그리고 shortcut rate/AvgSteps가 Control과 거의 동일
4. **애매 (seed 열지 않고 기록만)**: Perf만 오르고 Device/Misconfig는 위 바를 못 넘김 — "Link와의 경쟁"이 유일한 병목이 아님. §2 confound(노출량 증가)를 같이 기록

가설 지지만 되고 +5%p는 안 되면 seed 2/3는 열지 않고 기록만 한다.

9.9를 고려하면 **No-Go 또는 애매가 prior상 더 가능성 높은 결과**다. 이 실험의 가치는 mixture-level greedy shortcut 확인이다.

---

## 6. 체크리스트

- [x] 원본 terminal reward 완전 보존
- [x] 72-way action space 그대로 (축소하지 않음, link도 여전히 답으로 선택 가능한 옵션)
- [x] Gate / Factorized / Symmetric / E1 shaping / E2 timeout 전부 미적용
- [x] Episode generator에서 GT fault_type 샘플러만 {device, perf, misconfig} 균등으로 변경, link 배제
- [x] Control은 E3 zip 재사용 + **같은 NL eval 재평가** (25-ep 슬라이스 % 재사용 금지)
- [x] Primary eval: non-link-only 분포에서 stochastic + deterministic 둘 다
- [x] Secondary eval: 원래 4-fault 혼합 분포 (참고용, 판정에 안 씀)
- [x] §2의 confound(timestep 기준 노출량 약 1.33배)을 결과 문서에 명시
- [x] Evidence Acquisition을 rich(E1 predicate, eval 전용)와 simple(tool-use rate, E3 호환) 둘 다 기록
- [x] seed 1 판정 **no_go** — Device/Misconfig det 0/34·0/33, ΔNL +0%p, shortcut/AvgSteps 동일. seed 2/3 안 함

---

## 7. 이 실험이 답하는 것 / 답하지 않는 것

**답하는 것**: 4-way 혼합의 **greedy shortcut / AvgSteps**가 Link라는 값싼 경쟁 subtask를 빼면 흔들리는가. (및/또는 non-link 노출량이 늘면 Device/Misconfig의 0% 벽이 깨지는가 — prior는 낮음)

**답하지 않는 것**:
- Link와 non-link를 동시에 잘하는 단일 policy를 어떻게 만들 것인가 (배포 가능한 최종 policy는 이후 단계)
- "경쟁 제거"와 "노출량 증가" 중 어느 쪽이 더 크게 기여했는가 (§2, 후속 ablation 필요)
- specialist가 한 fault를 푸는가 (그건 9.9)
- 성공하더라도 E4(mask curriculum)가 불필요하다는 뜻은 아니다 — 4-fault 배포 policy는 결국 이 경쟁을 학습 전략으로 다뤄야 한다.
