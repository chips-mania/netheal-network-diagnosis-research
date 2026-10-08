# E1 — Pure Potential-Based Evidence Shaping 구현 스펙

## 0. 배경 (Cursor가 코드를 짜기 전에 알아야 할 것)

NetHeal (Star, 6 devices, 4 faults, MaskablePPO) 실험에서 다음 세 개입이 seed 1 기준으로 실패했다:

- **Symmetric partial reward**: DSR 19% → 4%로 악화
- **Diagnosis gate**: shortcut은 없앴지만 `scan → dummy ping → diagnose`로 형식만 만족
- **Factorized diagnosis (type→location)**: DSR 0%, timeout(-1.5)이 오진단(-16.1 리스크)보다 안전해서 아예 진단을 회피

공통 원인: **"증거를 실제로 모았을 때 → 정답에 가까워진다"는 dense credit이 없다.** 최종 결과(exact diagnosis)와 형식적 제약(도구 사용 여부)에만 보상이 걸려 있어서, PPO는 항상 가장 싼 경로로 수렴한다.

이번 E1은 이 문제를 **원래 reward/action 구조를 전혀 건드리지 않고, dense shaping 항 하나만 추가**해서 해결되는지 보는 실험이다. Gate/Factorized/Symmetric과 절대 섞지 않는다.

---

## 1. E1 = Control + 정확히 한 가지 항

| 구성 요소 | Control | E1 |
|---|---|---|
| Algorithm | MaskablePPO | MaskablePPO (동일) |
| Diagnosis action | 72-way (type×location) | 동일 |
| Terminal reward (exact/wrong/timeout) | 원본 그대로 | **원본 그대로, 수정 금지** |
| Diagnosis gate | 없음 | 없음 |
| Factorization | 없음 | 없음 |
| Symmetric reward | 없음 | 없음 |
| **Shaping term** | 없음 | **`r' = r + α·[γΦ(s') − Φ(s)]`** |

E1에서 바뀌는 건 딱 이 shaping term 하나뿐이어야 한다. 다른 항목을 하나라도 같이 바꾸면 "무엇 때문에 결과가 바뀌었는지" 알 수 없게 되므로 실험이 무효가 된다.

---

## 2. Φ(s) 정의

### 2.1 값의 의미

```
Φ(s) = 0   정답(type, location)을 구분할 relevant evidence를 아직 하나도 확보하지 못함
Φ(s) = 1   fault type 또는 location 중 한 축만 구분할 relevant evidence 확보
Φ(s) = 2   fault type과 location 두 축을 모두 구분할 relevant evidence 확보
```

- Φ = 2라고 해서 에이전트가 "정답을 안다"는 뜻이 아니다. **시뮬레이터 ground truth 기준으로, 정답을 식별 가능한 evidence가 지금까지의 history 안에 존재한다**는 뜻이다.
- 에이전트의 observation에는 ground truth가 절대 노출되지 않는다. Φ 계산에만 (training 코드 내부에서) simulator의 hidden fault state를 참조한다.
- 문서/README에 반드시 다음 문장을 그대로 남길 것:
  > "Simulator ground truth was used only to compute the training-time shaping potential; it was not exposed to the policy observation."

### 2.2 Φ를 절대 이렇게 정의하면 안 되는 것 (checkbox-gaming 재발 지점)

```python
# ❌ 금지: 도구 호출 여부로 Φ 계산
if used_ping: phi += 1
if used_traceroute: phi += 1
```
이러면 "도구 종류별로 한 번씩 누르면 Φ가 오른다"가 되어 Gate 실험과 같은 실패가 shaping 레벨에서 재발한다.

```python
# ✅ 올바름: 도구가 반환한 값이 실제 fault를 가리키는 이상 신호일 때만 Φ 증가
if tool_result.status == "down" or tool_result.reachability == "unreachable":
    # 이 evidence가 실제 groundtruth fault type과 일치하는 경우에만
    phi_type_axis = 1
```

핵심 규칙: **evidence 획득 여부가 아니라, 그 evidence가 실제 groundtruth와 일치하는 "diagnostic relevance"를 가졌는지**로 Φ를 올린다. 정상 신호(ping success, status up 등)를 받은 경우엔 도구를 썼어도 Φ가 오르면 안 된다.

### 2.3 Bounded / capped

Φ ∈ {0, 1, 2}로 상한을 둔다. 이유는 이론적 policy invariance 때문이 아니라 (그건 potential-difference 구조 자체가 담당) **reward scale 안정화와 pathological potential growth 방지**를 위한 것이다. 이 둘을 같은 이유로 혼동해서 설명하지 말 것.

---

## 3. Terminal 처리 (반드시 구현 전 감사 필요)

### 3.1 왜 필요한가

Discounted shaping reward를 episode 전체에 대해 telescoping하면:

```
Σ γ^t · F(s_t, s_t+1) = -Φ(s_0) + γ^T · Φ(s_T)
```

`Φ(s_0)`는 policy와 무관한 상수라 문제 없지만, **`γ^T·Φ(s_T)`는 policy-dependent**하다. 종료 시점(T)과 종료 상태의 Φ값이 policy가 선택한 행동에 따라 달라지기 때문에, 이 항을 방치하면 원래 reward 설계자가 의도한 correct/wrong/timeout 간 상대적 유불리가 왜곡될 수 있다.

### 3.2 규칙

```python
if next_state is genuinely terminal:
    phi_next = 0
else:
    phi_next = compute_phi(next_state)

shaping = alpha * (gamma * phi_next - phi_current)
```

### 3.3 Cursor가 코드 구현 전 반드시 확인해야 할 것

**NetHeal의 timeout이 `terminated`인지 `truncated`인지 먼저 확인할 것.**

- diagnose 액션으로 끝나는 episode → 명백한 `terminated` → Φ=0 처리 적용
- max_episode_steps(15) 도달로 끝나는 timeout → Gymnasium 관례상 `truncated`일 가능성이 있음

만약 코드상 timeout이 순수 `truncated=True` (단순 time-limit)로 구현되어 있다면, 이걸 diagnose-termination과 동일하게 Φ=0 처리하는 게 이론적으로 깔끔하다고 단정할 수 없다. **NetHeal이 timeout 자체를 "실패 episode"라는 실제 task outcome으로 정의하고 timeout penalty를 부여한다면, 이번 실험에서는 terminated와 동일하게 취급하고 Φ=0으로 두는 것이 일관적이다.** 이 판단을 내리기 전에:

1. env 코드에서 `step()`이 timeout 시 `terminated`/`truncated` 중 무엇을 반환하는지 확인
2. timeout penalty(-1.5)가 부여되는 코드 경로 확인
3. 위 둘이 확인된 후, 이 스펙 문서의 "genuinely terminal" 정의에 timeout을 포함할지 결정

이 항목은 **PPO 학습 실행 전에 체크리스트로 먼저 통과**시켜야 한다.

---

## 4. α (shaping scale factor)

- 반드시 **결과를 보기 전에 하나로 고정**한다. 여러 α를 실험해보고 best result를 골라 쓰는 것은 사실상 hyperparameter search이며, "PBRS 하나만 추가했다"는 실험의 순수성을 깨뜨린다.
- 기준: shaping reward의 최대 누적치가 원래 exact reward(+15.9)보다 항상 작아야 한다. 예: Φ가 0→2로 한 번에 오르는 극단적 케이스에서도 `α·γ·2` << 15.9가 되도록 설정.
- 고정값 선정 후 그 값과 선정 근거(계산식)를 결과 문서에 명시.

---

## 5. E1 체크리스트 (구현 순서)

- [ ] NetHeal 원본 reward 함수 완전 보존 확인 (exact/wrong/partial/timeout 값 그대로)
- [ ] 72-way diagnosis action 구조 변경 없음 확인
- [ ] Diagnosis gate 미적용 확인
- [ ] Factorization 미적용 확인
- [ ] Symmetric reward 미적용 확인
- [ ] timeout이 `terminated`/`truncated` 중 무엇으로 반환되는지 env 코드 감사
- [ ] Φ 계산 함수: tool 호출 여부가 아니라 반환된 evidence의 diagnostic relevance 기준으로 구현
- [ ] Φ 값 범위 {0, 1, 2}로 bound
- [ ] genuine terminal에서 Φ(next)=0 강제 로직 구현 (3.3 판단 반영)
- [ ] ground truth는 shaping 계산 내부 함수에서만 참조, policy observation에는 미노출 확인
- [ ] α 사전 고정 (결과 확인 전)
- [ ] Shaping reward unit test (아래 6번) 통과
- [ ] seed 1/2/3로 Control과 E1 각각 실행

---

## 6. Shaping Reward Unit Test (PPO 학습 전 필수 통과)

PPO를 돌리기 전에 아래 다섯 케이스를 코드 레벨에서 먼저 검증한다. 이걸 안 하면 나중에 PPO가 이상하게 움직였을 때 "아이디어가 틀렸나 구현이 틀렸나"를 구분할 수 없다.

| 케이스 | 상태 전이 | 기대되는 shaping |
|---|---|---|
| 1 | Φ: 0 → 0 (무의미한 ping, 정상 응답) | shaping ≈ 0 |
| 2 | Φ: 0 → 1 (relevant evidence 최초 획득) | positive shaping |
| 3 | Φ: 1 → 1 (같은 evidence 반복 확인/무관한 도구 재사용) | 새로운 큰 positive reward 없음 (≈0) |
| 4 | Φ: 1 → 2 (두 번째 축 evidence 확보) | positive shaping |
| 5 | Φ: 2 → terminal (diagnose로 episode 종료) | Φ_terminal = 0 적용, 이 transition의 shaping은 `-α·Φ(s_{T-1})` |

케이스 3이 특히 중요하다 — 여기서 큰 positive reward가 나온다면 checkbox-gaming이 shaping 레벨에서 재발했다는 신호이므로 Φ 정의를 다시 봐야 한다.

---

## 7. 결과 비교표 (Primary result용, seed 1/2/3 mean ± SD)

| Variant | Exact DSR | Type Acc | Loc Acc | Evidence Acquisition | Evidence-Consistent Diagnosis | Shortcut Rate |
|---|---|---|---|---|---|---|
| Masked Control | | | | | | |
| Symmetric Reward | | | | | | |
| Diagnosis Gate | | | | | | |
| Factorized | | | | | | |
| **E1 — Pure PBRS** | | | | | | |

### 지표 정의

- **Evidence Acquisition Rate**: diagnosis 제출 전에, 실제 fault를 구분할 수 있는 relevant evidence를 확보한 episode의 비율
- **Evidence-Consistent Diagnosis Rate**: relevant evidence를 확보한 episode 중에서, 그 evidence와 일치하는 최종 diagnosis를 제출한 비율
- 두 지표 모두 "PPO가 evidence를 인과적으로 이용했다"는 증명은 아니다. "관련 evidence를 확보한 뒤 그것과 일치하는 진단을 했다"는 관측적 사실만을 나타낸다. Counterfactual/ablation을 통한 causal claim은 이번 프로젝트 범위 밖.

### E1 성공/실패 해석 기준

**강한 성공 패턴 (예시)**:
```
DSR:                  19% → 44%
Evidence Acquisition: 18% → 71%
Evidence Consistency: 25% → 63%
Shortcut Rate:        85% → 22%
```
→ "reward를 바꿨더니 정확도가 올랐다"가 아니라, "evidence를 실제로 모으는 행동이 늘었고, 모은 evidence와 일치하는 진단도 늘면서 shortcut이 줄었다"고 말할 수 있는 근거가 된다.

**실패/의심 패턴 (예시)** — Φ가 checkbox처럼 게임당한 경우:
```
DSR:                  35%
Evidence Acquisition: 90%
Evidence Consistency: 20%
Shortcut Rate:        10%
```
→ tool은 많이 쓰지만 evidence와 최종 diagnosis가 연결되지 않음. Gate 실험에서 봤던 "형식적 준수" 패턴이 shaping 레벨에서 재발한 것으로 해석.

**DSR만 보고 판단 금지**: DSR 상승이 evidence-conditioned commit에서 온 것인지, 위 두 지표로 반드시 교차 확인한다.

---

## 8. 하지 말아야 할 것 (재확인)

- E1에 diagnosis gate, factorization, symmetric reward를 섞지 않는다
- 원본 terminal reward(exact/wrong/partial/timeout 수치)를 수정하지 않는다
- Link scan의 정보 비대칭(missing edge가 바로 보이는 것)을 인위적으로 완화하지 않는다 — NetHeal 환경의 특성으로 그대로 유지
- α를 결과 확인 후 소급 조정하지 않는다
- "PBRS라서/capped Φ라서 specification gaming이 이론적으로 불가능하다"는 과장된 주장을 쓰지 않는다. 정확한 표현:
  > "Potential-difference shaping으로 원래 objective와의 정렬을 유지하도록 설계하고, bounded Φ와 terminal Φ=0 처리를 통해 reward scale과 episodic boundary의 왜곡을 제한한다."
- seed 1 결과만으로 "Gate가 실패했다/Factorized가 실패했다"를 확정하지 않는다. 표현:
  > "개발 단계 seed 1에서 failure pattern을 관찰했고, 최종 비교에서는 3 seeds로 재검증한다."

---

## 9. E1 이후

- E1이 성공하면: dense credit assignment가 핵심이었다는 근거 → Primary 결과로 기록, 이후 100k scaling 고려
- E1이 실패하면: 문제가 dense reward 부재가 아니라 terminal reward 비대칭(exact/wrong/timeout 상대값)에 있다는 근거 → E2(graded diagnosis reward, dynamic timeout penalty)로 별도 진행. E1과 E2를 처음부터 섞지 않는다.
- Curriculum (type-only → location-only → joint, 또는 single-fault → 2-fault → 4-fault)은 E1/E2 이후 별도 variant로 고려. 지금 먼저 하면 "reward 문제를 회피해서 쉬운 문제부터 가르친 것"과 "credit assignment를 고친 것"이 섞여 해석 불가능해진다.
- Star/Linear → Mesh 일반화 검증은 위 전부가 끝난 뒤 마지막 optional validation.
