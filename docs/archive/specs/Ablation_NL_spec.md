# Ablation-NL — Non-Link-Only 학습 분포 실험 스펙

## 0. 배경 및 질문

Control deterministic Exact DSR(20%)이 Link DSR(24%, episode의 약 1/4)과 거의 일치한다. 즉 aggregate 정확도의 거의 전부가 Link 하나에서 나오고, Device/Perf/Misconfig는 즉시-진단 경로에서 사실상 기여가 없다.

가설: **Link가 "scan 후 즉시 진단"만으로 완벽히 풀리는 값싼 subtask라서, 매 episode 학습 신호에 계속 경쟁 상대로 끼어들어 나머지 세 fault(더 어렵고, 정책 복잡도가 높은)에 대한 학습 신호를 구조적으로 억누른다.**

E1(dense shaping)/E2(timeout 비대칭)/E3(exploration) 모두 이 구조를 그대로 둔 채 aggregate reward 안에서 뭔가를 조정했다. 만약 진짜 병목이 "Link와의 경쟁"이라면, aggregate 안에서 뭘 바꿔도 이 쏠림 자체는 안 없어진다.

이 실험은 **reward도 action space도 observability도 안 건드리고, 학습 시 fault 샘플링 분포에서 Link를 제외**해서 이 가설을 직접 확인한다. E1/E2/E3와 섞지 않는다.

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

**중요**: action space나 diagnosis 로직은 그대로 둔다. 학습용 episode generator에서 GT fault_type을 뽑는 샘플러 하나만 바꾼다. Link의 scan observability(missing edge가 바로 보이는 것)는 전혀 안 건드린다 — 이건 학습 분포 조정이지 환경 수정이 아니다.

---

## 2. 알려진 confound (결과 해석 시 반드시 명시)

Control에서 non-link 세 fault는 전체 episode의 약 3/4(각 25%)만 차지했다. Ablation-NL에서는 전체 episode의 100%가 non-link 세 fault로, 각 fault가 받는 학습 노출량이 **약 1.33배 늘어난다** (30k episode 기준 각 fault ~7.5k → ~10k).

즉 이 실험에서 non-link DSR이 오른다면, 그게 **(a) Link와의 경쟁이 사라져서**인지 **(b) 단순히 각 fault의 학습 데이터가 늘어나서**인지 이 ablation 하나로는 분리가 안 된다. 이건 알고 시작하는 한계로 명시하고, 두 원인 다 "Link가 구조적으로 non-link 학습을 방해한다"는 큰 그림과는 상충하지 않으므로 1차 검증으로는 충분하다. 원인을 더 분리하고 싶다면 후속으로 "Link 포함하되 빈도만 줄인 버전"(예: link 10%, 나머지 30%씩)을 별도로 돌려볼 수 있다 — 지금 단계에서는 안 한다.

---

## 3. 평가 (Eval)

### 3.1 Primary eval — non-link-only 분포

학습 분포와 동일하게 {device, perf, misconfig}만으로 eval episode 생성. Control과 비교하려면 **Control 쪽도 non-link fault episode만 슬라이스**해서 비교해야 공정하다 (Control이 이미 갖고 있는 per-fault 분해 결과를 재사용).

### 3.2 Secondary eval — 원래 4-fault 혼합 분포 (참고용, 판정에 안 씀)

Ablation-NL로 학습한 모델을 원래 {link, device, perf, misconfig} 혼합 분포에서 eval. Link에 대해 어떻게 반응하는지 관찰용 (예: link episode에서도 여전히 non-link 답을 내는지, 우연히 맞는 비율이 얼마인지). 이건 이 실험의 성패 기준이 아니라 부가 관찰이다.

### 3.3 Stochastic + Deterministic 둘 다 (E3와 동일 관례)

E3 리포트 형식을 따라 stochastic policy와 greedy(deterministic) mode 둘 다 기록한다 — E3에서 stochastic과 deterministic이 완전히 다른 그림을 보여줬으므로 이번에도 필수.

---

## 4. 지표 표

### 4.1 Primary (seed 1, stochastic + deterministic)

| Metric | Control (non-link slice, stochastic) | Control (non-link slice, det) | Ablation-NL (stochastic) | Ablation-NL (det) |
|---|---|---|---|---|
| Non-link DSR | | | | |
| Device DSR | | | | |
| Perf DSR | | | | |
| Misconfig DSR | | | | |
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

## 5. 판정 기준

**성공 (가설 지지)**: non-link DSR(device/perf/misconfig 평균 또는 개별)이 Control의 non-link slice 대비 **의미 있게 상승** — 특히 Device/Misconfig처럼 지금까지 0%였던 fault에서 0%를 벗어나는지가 핵심 신호. Evidence-Consistent Diagnosis도 같이 오르는지 확인 (오르지 않으면 "우연히 다른 shortcut으로 대체된 것"일 수 있으므로 반드시 같이 본다).

**실패 (가설 기각)**: Link를 완전히 제거했는데도 non-link DSR이 거의 안 움직이고, Device/Misconfig가 여전히 0%에 가까움. 이 경우 문제는 "Link와의 경쟁"이 아니라 **evidence-gathering + 72-way joint classification 자체가 (경쟁 유무와 무관하게) 지금 학습 세팅에서 원래 어렵다**는 뜻이 되고, E4(diagnose-mask curriculum)가 그대로 다음 순서가 된다.

**애매한 경우**: 일부만 오름 (예: Perf만 오르고 Device/Misconfig는 그대로). 이 경우 "Link와의 경쟁"이 유일한 병목이 아니라 fault별로 별도의 난이도 요인이 있다는 뜻 — fault별 evidence 조건(§2.4 of E1 spec 참조)의 난이도 차이를 다시 봐야 한다.

**Go/No-Go**: seed 1에서 non-link DSR 상승폭 기준을 사전에 하나 정해둔다 (예: +5%p 이상이면 Go → seed 2/3, 미만이면 No-Go, exploratory로만 기록). 정확한 임계값은 Control의 seed-1 variance 참고 자료가 있으면 그걸로 잡고, 없으면 이전 개입들과 동일하게 **+3%p**를 기본선으로 쓴다 (E1/E2에서 쓴 기준과 일관성 유지).

---

## 6. 체크리스트

- [ ] 원본 terminal reward 완전 보존
- [ ] 72-way action space 그대로 (축소하지 않음, link도 여전히 답으로 선택 가능한 옵션)
- [ ] Gate / Factorized / Symmetric / E1 shaping / E2 timeout 전부 미적용
- [ ] Episode generator에서 GT fault_type 샘플러만 {device, perf, misconfig} 균등으로 변경, link 배제
- [ ] Control 쪽 non-link slice 비교 데이터 준비 (기존 결과에서 재사용 가능한지 확인, 없으면 재계산)
- [ ] Primary eval: non-link-only 분포에서 stochastic + deterministic 둘 다
- [ ] Secondary eval: 원래 4-fault 혼합 분포 (참고용, 판정에 안 씀)
- [ ] §2의 confound(노출량 1.33배 증가)을 결과 문서에 명시
- [ ] seed 1 Go/No-Go 판정 (+3%p 기준) → Go 시 seed 2/3

---

## 7. 이 실험이 답하는 것 / 답하지 않는 것

**답하는 것**: Link라는 값싼 경쟁 subtask를 제거하면(및/또는 non-link 노출량이 늘면) non-link troubleshooting 학습이 실제로 열리는가.

**답하지 않는 것**:
- Link와 non-link를 동시에 잘하는 단일 policy를 어떻게 만들 것인가 (이건 이 ablation의 범위 밖 — 배포 가능한 최종 policy를 만드는 문제는 이후 단계)
- "경쟁 제거"와 "노출량 증가" 중 어느 쪽이 더 크게 기여했는가 (§2, 후속 ablation 필요)
- 성공하더라도, 이게 곧 E4(mask curriculum)가 불필요하다는 뜻은 아니다 — Link를 다시 포함한 4-fault 환경에서 배포 가능한 policy를 만들려면 결국 이 경쟁 구조를 학습 전략으로 다뤄야 하고, 그때 mask curriculum이나 다른 방법이 다시 필요할 수 있다.
