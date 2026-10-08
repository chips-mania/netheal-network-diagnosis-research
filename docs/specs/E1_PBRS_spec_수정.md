# E1 — Pure Potential-Based Evidence Shaping 구현 스펙

## -1. 리뷰 반영 (v4)

v3까지 Φ 스키마·α/γ·non-link 지표를 고정했다. v4는 location 정의 대조와 Go/No-Go 숫자를 닫는다.

1. **E1은 필요조건 실험이지 해결책이 아니다.** Φ가 terminal에서 0으로 회수되면 episode 전체 shaping 합은 상수(≈0)이고, exact/wrong/timeout 상대값은 그대로다. 묻는 것은 “중간 credit을 당기는 것만으로 non-link evidence-seeking이 느는가”이다.
2. **Φ lookup은 초안이 아니라 확정 predicate** (2.4). NetHeal 시뮬레이터에 `blocked`/`misconfigured` 필드는 없다. misconfig는 연결 하나를 `down`으로 끄는 구현이다.
3. **Link는 Φ 크레딧에서 배제.** link episode의 **reward는 Control과 동일**하다. 공유 정책이라 non-link shaping 때문에 link **행동**까지 동일하다고 단정하지 않는다.
4. **γ(shaping) = PPO discount** (SB3 기본 0.99). **α = 1** 사전 고정.
5. **timeout**: `terminated`(diagnose) / `truncated`(15 step). 둘 다 Φ(next)=0. clawback은 감시 지표로만 남긴다 (7.1).
6. **Evidence 지표는 non-link episode만 집계.** link는 정의상 Φ=0이라 aggregate를 희석한다.
7. **성공 기준**: non-link DSR + Evidence Consistency. Link만 오르면 실패.
8. **Primary는 Control vs E1.** seed 1 Go/No-Go는 7.2 숫자 기준. 통과 시에만 seed 2/3. R/G/F는 seed-1 exploratory.
9. E1은 **training-time oracle potential** (privileged shaping). observation에는 GT를 넣지 않는다.
10. **Misconfig GT location은 device** (`injector.py`: `FaultInfo(..., f"{device_id}", {blocked_destination})`). 72-way exact match도 device 문자열. location축은 edge 쌍이 아니다.
11. Unit test **case 8**: GT=misconfig + scan only → Φ=0.
12. **Link DSR Δ는 secondary sanity** (pass/fail 아님). shared-policy spillover 감시용.

---

## 0. 배경

NetHeal (Star, 6 devices, 4 faults, MaskablePPO)에서 다음 세 개입이 seed 1 기준으로 실패했다.

- **Symmetric partial reward**: DSR 19% → 4%
- **Diagnosis gate**: shortcut은 사라졌지만 `scan → dummy ping → diagnose`
- **Factorized diagnosis**: DSR 0%. timeout(−1.5)이 오진단(−16.1)보다 안전해서 진단을 회피

공통점: 증거를 모았을 때 정답에 가까워진다는 **dense credit이 없다.** PPO는 가장 싼 경로로 수렴한다.

E1은 원본 reward/action을 건드리지 않고 shaping 항만 추가해서, **앞당긴 credit만으로 non-link evidence-seeking이 느는지를 보는 실험**이다. Gate / Factorized / Symmetric과 섞지 않는다. 해결책이라고 쓰지 않는다.

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
| Shaping 정보 | 없음 | **training-time oracle potential** (GT는 reward 계산에만 사용) |
| **Shaping term** | 없음 | **`r' = r + α·[γΦ(s') − Φ(s)]`** |

E1에서 바뀌는 것은 이 shaping term 하나다. 다른 항목을 같이 바꾸면 실험이 무효다.

문서/README에 다음 문장을 남긴다.

> "Simulator ground truth was used only to compute the training-time shaping potential; it was not exposed to the policy observation."

Eval은 원본 NetHeal reward로 한다 (shaping off 또는 동일 wrapper라도 메트릭은 원본 success criterion).

---

## 2. Φ(s) 정의

### 2.1 값의 의미

```
Φ(s) = 0   type/location을 구분할 relevant evidence 없음
Φ(s) = 1   type 또는 location 중 한 축만 확보
Φ(s) = 2   type과 location 두 축 모두 확보
```

Φ=2는 에이전트가 정답을 안다는 뜻이 아니다. **hidden GT 기준으로, 그 정답을 식별 가능한 evidence가 history에 있다**는 뜻이다. observation에는 GT를 넣지 않는다.

### 2.2 금지 (checkbox-gaming)

```python
# ❌ 도구 호출 여부
if used_ping: phi += 1
```

정상 응답(ping success + 낮은 latency, status up, interface 전부 정상)으로는 Φ를 올리지 않는다.

### 2.3 Bounded

Φ ∈ {0, 1, 2}. policy invariance 때문이 아니라 reward scale 안정화와 potential 폭주 방지다.

### 2.4 Fault별 Φ predicate (확정 — 학습 전 이 표만 구현)

**Link (`link_failure`)**: Φ(s) = 0 고정. scan으로 missing edge가 type+location을 드러내므로, 여기에 크레딧을 주면 기존 shortcut에 보너스를 얹는다.  
link episode의 **reward는 Control과 같다.** 4-fault 공유 정책이라 link **행동**까지 같다고 쓰지 않는다.

**Misconfig 구현 사실**: `inject_misconfiguration`은 방화벽 device에서 연결 하나를 `down`으로 끈다 (`blocked_destination`). 도구는 `"misconfigured"`를 반환하지 않는다. ping은 unreachable / no route, interface는 `status==down`, scan은 missing cell(link처럼 보임). **scan만으로는 misconfig 축을 올리지 않는다.** type 크레딧은 GT==misconfiguration일 때만 준다 (원시 신호가 link와 같기 때문).

**Misconfig location은 device다 (코드 대조, 확정).**  
`FaultInfo`의 location 필드는 `f"{device_id}"`이고, `blocked_destination`은 `details`에만 있다. 72-way에서 misconfig 진단은 device 슬롯 6개(device_failure와 동일)다. exact DSR / `_locations_match`는 `predicted_location == device_id` (문자열 일치). `firewall->blocked_dest` 형태를 제출하면 오답이다.  
따라서 location축은 **그 device를 가리키는 evidence**이지, blocked edge의 양 끝점을 모두 맞추는 것이 아니다. ping dest가 `blocked_destination`이기만 하고 GT device가 아니면 location축을 올리지 않는다.

**Perf**: `check_status`는 정상(up)일 수 있다. status만으로 type축을 올리지 않는다.

아래 문자열/필드는 `netheal/tools/simulator.py`와 9.11 실측값에 맞춘다.

| GT | type축 (1 또는 0) | location축 (1 또는 0) |
|---|---|---|
| `link_failure` | 올리지 않음 | 올리지 않음 |
| `device_failure` | `ping`의 `data.error`에 `unreachable` / `No route to host` / `is down` **또는** `check_status`의 `data.status == "down"` | 그 ping dest 또는 status device가 GT location과 일치 |
| `performance_degradation` | `check_interfaces`에서 한 interface의 `latency_ms`가 **이상** (아래 임계값). `ping`/`traceroute`가 `latency_ms` / `total_latency_ms`로 같은 임계값을 넘어도 type축 인정. **status만으로는 불가** | 이상 latency가 난 interface의 `destination`이 GT link의 한 끝과 일치 (GT `center->edge_0`이면 dest가 `edge_0` 또는 `center`) |
| `misconfiguration` | scan 제외. `ping` error가 unreachable/no route/down **또는** `check_interfaces`에서 한 interface `status == "down"`. **그리고 GT type == misconfiguration** | **device 매칭만.** `ping` dest == GT `device_id` **또는** `check_interfaces(device_id)`에서 그 device == GT이고 한 interface가 `down`. blocked edge 쌍(`device->blocked_destination`)을 location으로 인정하지 않음 |

**Perf latency 임계값 (사전 고정, 소급 변경 금지)**  
9.11 Star 실측: 정상 interface ≈ 1.4–3.8ms, 고장 링크 ≈ 24 / 38 / 61ms.

이상 판정 (둘 중 하나면 충분):

- 같은 `check_interfaces` 결과 안에서 `latency_ms >= 5 × (나머지 interface latency의 median)`
- 또는 `latency_ms >= 15.0`

한 device에 interface가 1개뿐이면 15.0ms 기준만 쓴다.

**Φ 합산**

```
type_bit = 1 if type축 조건이 history 안에 한 번이라도 참
loc_bit  = 1 if location축 조건이 history 안에 한 번이라도 참
Φ(s) = 0                         if GT == link_failure
     = type_bit + loc_bit        otherwise   # {0,1,2}
```

같은 evidence를 반복해도 bit는 그대로다 (케이스 3).

**한 스텝 0→2**  
device에서 `ping(GT device)`가 unreachable이면 type과 location이 동시에 참이 된다. 허용한다. α·γ·2 ≈ 1.98.

---

## 3. Terminal 처리

### 3.1 Telescoping

```
Σ γ^t · F(s_t, s_t+1) = -Φ(s_0) + γ^T · Φ(s_T)
```

`Φ(s_T)`를 방치하면 correct/wrong/timeout 상대값이 왜곡된다.

### 3.2 규칙

```python
if terminated or truncated:
    phi_next = 0
else:
    phi_next = compute_phi(next_state)

shaping = alpha * (gamma * phi_next - phi_current)
```

### 3.3 Timeout — 코드 확인 완료 (`env.py` 278–280)

```python
terminated = self._is_terminated(action_spec)   # diagnose
truncated = self.step_count >= self.max_episode_steps
```

- 별도 timeout penalty 없음. −1.5는 −0.1 × 15.
- **결정:** `terminated`와 `truncated` 모두 Φ(next)=0.

**알려진 부작용 (스펙을 더 고치지 않음):**  
evidence를 모은 뒤 timeout하면 clawback `−α·Φ ≈ −2`가 추가된다. 안 모으고 timeout하는 편이 싸다. Factorized 회피가 약하게 남을 수 있다. 7.1에서 전반/후반 Evidence Acquisition을 나눠 본다.

---

## 4. α와 γ

- **α = 1** (결과 본 뒤 변경 금지)
- **γ = MaskablePPO discount** (기본 0.99). config에서 바꿨으면 그 값에 맞춘다. 다르면 PBRS가 아니다.
- 0→2 한 번에 `α·γ·2 ≈ 1.98` ≈ exact(+15.9)의 1/8.
- α 그리드 서치 금지.

---

## 5. 체크리스트

- [ ] 원본 reward 보존 (exact/wrong/partial/step)
- [ ] 72-way diagnosis 변경 없음
- [ ] Gate / Factorized / Symmetric 미적용
- [ ] timeout = `terminated` / `truncated` 확인
- [ ] Φ는 2.4 predicate만 사용. 도구 호출 횟수 금지. `misconfigured` 필드 발명 금지
- [ ] Φ ∈ {0, 1, 2}, link는 항상 0
- [ ] terminal/truncated에서 Φ(next)=0
- [ ] GT는 shaping 내부만, observation 미노출
- [ ] α=1, γ=PPO γ
- [ ] unit test 1–8 통과 후에만 PPO
- [ ] seed 1 Control vs E1 — 7.2 숫자 Go/No-Go 통과 시에만 seed 2/3

---

## 6. Shaping unit test (PPO 전 필수)

| # | 전이 | 기대 |
|---|---|---|
| 1 | Φ 0→0 (정상 ping / status up) | shaping ≈ 0 |
| 2 | Φ 0→1 (한 축만 최초 획득) | positive ≈ `α(γ·1 − 0)` |
| 3 | Φ 1→1 (같은 evidence 반복) | ≈ 0 |
| 4 | Φ 1→2 (둘째 축) | positive ≈ `α(γ·2 − 1)` |
| 5 | Φ 2→terminal (diagnose) | `−α·Φ(s_{T-1})` = −2 |
| 6 | **GT=link**, scan 후 diagnose | **전 구간 shaping ≈ 0** |
| 7 | **GT=device**, `ping(GT device)` unreachable | **0→2 한 스텝 허용**, positive ≈ `α·γ·2` |
| 8 | **GT=misconfiguration**, scan만 (missing/down link처럼 보임). ping/`check_interfaces` 없음 | **Φ=0 유지.** scan만으로는 type/location축을 올리지 않음 |

3번에서 큰 plus가 나오면 checkbox 재발. 6번이 새면 Link 배제가 깨진 것이다. 8번이 새면 misconfig를 scan-shortcut으로 크레딧한 것이다.

---

## 7. 결과 비교표

### 7.0 Primary (Control vs E1)

학습: Star, 6 devices, 4 faults, MaskablePPO, 30k, `max_episode_steps=15`.  
먼저 seed 1. **7.2 Go/No-Go**를 통과하면 seed 1/2/3 mean ± SD. 실패면 seed 2/3를 하지 않는다.

| Variant | Exact DSR | **Non-link DSR** | Type Acc | Loc Acc | Evidence Acquisition (non-link) | Evidence-Consistent Diagnosis (non-link) | Shortcut Rate |
|---|---|---|---|---|---|---|---|
| Masked Control | | | | | | | |
| **E1 — Pure PBRS** | | | | | | | |

Fault별 DSR (Link만 오르고 나머지가 그대로면 실패):

| Variant | Link DSR | **Link DSR Δ** (sanity) | Device DSR | Perf DSR | Misconfig DSR |
|---|---|---|---|---|---|
| Masked Control | | — | | | |
| E1 — Pure PBRS | | | | | |

**Link DSR Δ**는 pass/fail이 아니다. link episode reward는 Control과 같지만 정책 파라미터는 공유되므로, non-link shaping이 link 행동에 간접으로 새는지를 보는 secondary sanity다. 크게 흔들리면 별도 기록하고, 그 이유만으로 E1을 성공/실패로 뒤집지 않는다.

### 7.0-b Exploratory (seed 1, Primary 아님)

| Variant (seed 1) | Exact DSR | Non-link DSR | Shortcut Rate |
|---|---|---|---|
| Symmetric Reward | | | |
| Diagnosis Gate | | | |
| Factorized | | | |

### 지표

- **Evidence Acquisition (non-link)**: non-link episode 중, diagnose 전에 Φ≥1이 된 비율. link episode는 분모에서 제외.
- **Evidence-Consistent Diagnosis (non-link)**: 위 확보 episode 중, 최종 diagnosis가 그 evidence와 맞는 비율 (type+location exact, 또는 최소한 확보한 축과 일치).
- 둘 다 causal claim이 아니다. “관련 evidence 뒤에 그것과 맞는 진단을 했다”는 관측이다.

### 7.1 성공/실패

Aggregate Exact DSR만으로 판정하지 않는다. 성공은 아래를 **모두** 만족할 때다.

1. **Non-link DSR**이 Control보다 의미 있게 상승
2. **Evidence-Consistent Diagnosis (non-link)**가 Control보다 상승 (acquisition만 오르면 Gate형 형식 준수)
3. **Link DSR만 오르고 Device/Perf/Misconfig가 그대로인 패턴이 아님**

참고용 예시 (실측 아님):

```
Non-link DSR:         12% → 35%
Evidence Acquisition: 18% → 71%
Evidence Consistency: 25% → 63%
Shortcut Rate:        85% → 22%
```

의심 (checkbox):

```
Non-link DSR:         14%
Evidence Acquisition: 90%
Evidence Consistency: 20%
```

**Clawback 감시:** Evidence Acquisition을 episode 전반/후반으로 나눠, step 15 근처에서 evidence-seeking이 죽는지만 본다. 있어도 이번 스펙은 바꾸지 않고 E2 근거로 남긴다.

### 7.2 Seed 1 Go/No-Go (사전 고정 숫자)

seed 1 결과를 보고 “움직인 것 같다”고 주관 판단하지 않는다. Control 대비 E1 (같은 eval suite):

```
No-Go  (둘 다 만족):
  Non-link DSR Δ            < +3%p
  Evidence Acquisition
    (non-link) Δ            < +5%p

Go:
  위 No-Go가 아니면 seed 2/3를 진행한다.
```

Δ는 `E1 − Control` (seed 1). 단위는 percentage point.  
예: Control non-link DSR 4%, E1 6% → Δ = +2%p → 첫 조건 해당. Acquisition이 +4%p면 No-Go.

No-Go면 seed 2/3를 돌리지 않고 E1 실패(dense credit만으로는 부족)로 기록한 뒤 E2를 검토한다.  
Go는 “성공”이 아니다. 3-seed Primary를 열겠다는 뜻이다. 최종 성공/실패는 7.1이다.

---

## 8. 하지 말 것

- Gate / Factorized / Symmetric을 E1에 섞지 않는다
- 원본 terminal/partial/step 수치를 수정하지 않는다
- Link scan observability를 환경에서 가리지 않는다 (Φ에서만 뺀다)
- α, latency 임계값(5× median / 15ms), **7.2 Go/No-Go 숫자(+3%p / +5%p)**를 결과 본 뒤 바꾸지 않는다
- “PBRS라서 gaming이 불가능하다”고 쓰지 않는다. 쓸 문장:
  > "Potential-difference shaping으로 원래 objective와의 정렬을 유지하도록 설계하고, bounded Φ와 terminal Φ=0 처리를 통해 reward scale과 episodic boundary의 왜곡을 제한한다."
- seed 1만으로 R/G/F 실패를 Primary 결론으로 쓰지 않는다:
  > "개발 단계 seed 1에서 failure pattern을 관찰했고, Control vs E1 최종 비교는 통과 시 3 seeds로 한다."

---

## 8-b. E1이 답하는 것 / 답하지 않는 것

**답하는 것:** 72-way와 terminal reward를 유지한 채, evidence에 dense credit을 앞당기는 것만으로 PPO가 도구를 더 쓰고 non-link를 더 맞히는가.

**답하지 않는 것:**

- exact / wrong / timeout 리스크 비대칭 (Factorized 회피의 원인) → E2
- 완성된 troubleshooting policy
- reward **총량**을 바꾼 효과. 성공해도 “credit **타이밍**”의 효과라고 쓴다.

---

## 9. E1 이후

- 성공: dense credit이 병목이었다는 근거 → Primary 기록. 100k는 그 다음.
- 실패: 병목이 terminal 비대칭 쪽일 가능성 → E2 (graded diagnosis / timeout)만 따로. E1과 섞지 않는다.
- Curriculum은 E1/E2 이후.
- Star/Linear → Mesh는 그 다음 optional validation.
