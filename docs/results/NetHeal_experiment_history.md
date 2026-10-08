# NetHeal RL / Troubleshooting Protocol — 실험 전체 정리

**작성일:** 2026-09-06  
**목적:** 지금까지 돌린 감사·PoC·변형 실험을 나중에 다시 읽을 수 있게 한곳에 모은다.  
**코드:** `netheal-topology-generalization/`  
**환경(수정 금지):** `netheal-ai-agent-benchmark/` (Cisco NetHeal)  
**Conda:** `netheal_base`  
**세부 리포트:** `netheal-topology-generalization/reports/`  
**짧은 진행 로그:** `reports/step_progress.md`

---

## 0. 연구 질문과 범위

### 우리가 묻는 것 (Cisco 공식 과제와 구분)

> NetHeal의 **공식 RL 인터페이스**(Gymnasium + 원본 보상 + 진단 액션) 위에서,  
> **어떤 학습 프로토콜**이 실제 troubleshooting 정책(도구로 증거 수집 → 타입×위치 진단)을 만드는가?

Cisco가 공개한 것은 Gymnasium + vanilla PPO 예시(`docs/guides/training-sb3.md`)이다.  
대회/리더보드 주 경로는 AgentBeats/LLM + MCP이다. **Cisco가 PPO-DSR 리더보드를 발표한 것은 아니다.**

Mesh 일반화(Seen→Unseen)는 **Seen에서 유효한 troubleshooting 정책이 생긴 뒤**로 미룬다.

### 고정 실험 설정 (대부분의 PoC)

| 항목 | 값 |
|---|---|
| Topology | Star, 6 devices |
| Fault types | link / device / performance / misconfiguration (균등) |
| Algo | MaskablePPO (`sb3_contrib`), `MultiInputPolicy` (예외 명시) |
| Timesteps (RL) | 보통 **30k** |
| Seed | 주로 **1** (2/3는 대부분 미실행) |
| Eval | **100 episodes**, deterministic(greedy) 기본; 일부 stochastic |
| Max steps | 15 |
| Cisco 패키지 | **수정하지 않음** — 래퍼/프로토콜만 실험 레포 |

### 공식 보상 (Cisco, 변경 시 “named variant”)

출처: `reports/reward_structure.md`, `netheal/environment/rewards.py`

| 이벤트 | 대략값 (6 devices) |
|---|---|
| Exact 진단 성공 | **+16** (`10 * (1 + n/10)`) |
| Wrong 진단 | **−16** |
| Partial (location overlap 등) | ~+4.8 (type-only partial **없음**) |
| Step | **−0.1** |
| Invalid action | −5 |
| Timeout (진단 없이 trunc) | 스텝 페널티만 (원본). E2에서만 −16로 바꿈 |

**중요:** Exact 보너스는 **도구를 썼는지와 무관**하다. 추측이 맞아도 +16.  
공식 DSR은 guess-correct를 포함한다. “learned troubleshooting”을 말할 때는 별도로 본다:

- shortcut rate (`scan → diagnose`, ≤2 steps)
- empty-obs / 증거 없는 진단
- `exact_consistent` (증거 비트 Φ와 타입·위치가 맞음)

보상·마스크·샘플러를 바꾸면 **Cisco Control과 다른 과제**다. 이름은 반드시 variant로 남긴다.

### 진단 공간

- Star에서 diagnosis ≈ **72-way** = fault_type × location (Cartesian)
- Post-scan 직후에도 72개 진단이 즉시 valid
- Type만 맞추는 partial reward 없음

### Gym 관측 구멍 (9.10 / stochastic audit)

시뮬레이터 **툴 결과**에는 type/location 증거가 있다.  
Gym `to_dict()`는 일부를 버린다:

| 내부 상태 | Gym 벡터 |
|---|---|
| Device DOWN | `device_status[:,0]`가 `status=='active'` 문자열 비교 → **항상 0**에 가까움 |
| ping latency | `connection_properties`에만 있고 `discovery_matrix`에는 없음 |
| ping fail 문구 | `recent_diagnostics`에 success/hash 정도만 |

실험 레포 `src/evidence_obs.py` 래퍼가 같은 키·shape에 DOWN/latency를 채워 넣는다.  
**새 obs 키는 추가하지 않음.**

---

## 1. 한눈에 보는 결론

| 단계 | 결과 | 한 줄 |
|---|---|---|
| Vanilla PPO | invalid-diagnosis collapse | 마스크 없이 무효 진단으로 붕괴 |
| MaskablePPO 30k | Link 숏컷 | invalid 고정, `scan→diagnose`, Link만 상대적 고득점 |
| Link-only 30k | DSR ~100% | 스캔만으로 링크 위치가 갈림 |
| 보상/게이트/팩토라이즈/PBRS/timeout/entropy/NL제거 | 대부분 **no_go** | 숏컷·더미툴·타임아웃·회피 중 하나 |
| Obs 래퍼 Stage B | **no_go** | 칸은 보이는데 diagnose P(correct) 안 움직임 → E4 미실시 |
| Evidence diagnose-gate | **no_go** | shortcut 0%, timeout ~76% |
| Short Rule-BC seed + gate PPO | **ambiguous** | seed Exact 0% (미모방); gate 후 Exact 2%, timeout 50% |
| Full Rule-BC + gate PPO (pad6) | **go** | BC 98% → gate-ZS 48% → PPO Exact **77%**; Misconfig 약함 |
| Repeat penalty (−2) | **ambiguous** | 제출은 늘지만 Device/Misconfig cons 부족 |
| BC GT-전문가 3-step | Exact **58%** | Link/Misconfig만; Device/Perf 위치 모달 붕괴 |
| Rule ceiling (GT 없음) | Exact **100%** | 조건식 트리로 닫힘 |
| **BC on Rule 궤적** | Exact **99%**, **go** | 같은 MLP로 트리를 배울 수 있음; 병목은 데모 품질 |

**최종 해석 (Star Seen):**

1. 문제는 “분류 불가능”이 아니다. Rule이 GT 없이 100%.
2. 원본 보상 + MaskablePPO 30k는 troubleshooting이 아니라 Link 숏컷을 학습한다.
3. GT를 아는 3-step 데모 BC는 Device/Perf 바인딩을 못 복제한다.
4. **겹침 해소 트리(Rule) 궤적 BC**면 Device/Perf까지 99%까지 간다.
5. 병목은 “72-way가 본질적으로 불가”가 아니라 **데모/프로토콜이 트리를 담느냐**다.

---

## 2. 용어

| 용어 | 의미 |
|---|---|
| **DSR / Exact** | 타입+위치 완전 일치 비율 (Cisco Diagnosis Success Rate) |
| **shortcut** | `scan` 직후(또는 ≤2 step) 진단, 중간 diagnostic 툴 없음 |
| **Φ / evidence bits** | PBRS용 type/loc 증거 비트 (GT와 대조해 채점; 정책 입력에 GT 넣지 않음) |
| **exact_consistent** | exact이면서 Φ·타입·위치가 증거와 일치 |
| **GT** | ground truth fault (시뮬레이터가 아는 정답). **채점**에는 쓰고, **Rule/정책 액션**에는 안 씀 |
| **GT-free Rule** | 액션 선택에 GT를 쓰지 않는 규칙 에이전트 |
| **BC-informed** | GT 전문가 `scan→증거툴 1개→정답진단` 데모 모방 |
| **BC-rule** | Rule 궤적(겹침 시 추가 툴) 모방 |
| **named variant** | 보상/마스크/샘플러가 Cisco Control과 다른 실험 |

---

## 3. 초기 감사· Sanity (Step ~9.x)

상세: 각 `reports/*.md`. 여기만 요약.

### 3.1 Vanilla PPO → MaskablePPO

- **Vanilla PPO:** invalid diagnosis로 붕괴 (`engineering_note_ppo_degeneracy.md` 등).
- **Action mask:** invalid 제거. MaskablePPO Star 30k.
- **결과 (masked sanity):** DSR ~17%, AvgSteps **2.0**, shortcut성 `scan→diagnose`, Link DSR ~68%, Device/Perf/Misconfig ~0.  
  → Troubleshooting GO 실패 (`masked_ppo_sanity_gonogo.md`).

### 3.2 Link-only

- Link만 학습 30k → DSR **~100%**.
- Post-scan discovery에서 끊긴 간선이 보여 **스캔만으로 위치 가능**.
- Multi-fault에서도 정책이 Link 패턴으로 붕괴하기 쉬운 이유.

### 3.3 Post-scan information (9.8)

| Fault | 스캔 후 다른 타입과 분리? |
|---|---|
| link_failure | **Yes** (missing edge) |
| device / perf / misconfig | **No** (서로 비슷) |

스캔만으로 타입 probe ≈ 50% = link vs non-link만.

### 3.4 Non-link single-fault (9.9)

D/P/M 각각 단독 학습해도 “유효 troubleshooting 정책” GO 실패.  
단, **unsolvable 증명은 아님** (이후 Rule 100%로 확정).

### 3.5 Tool solvability / failure mode (9.10–9.11)

- 툴 결과만 보면 타입+위치 구분 가능 (Cisco `fault-behaviors.md`와 일치).
- PPO는 그 툴을 안 쓰거나, 써도 진단을 안 맞춤.
- Misconfig: 위치는 읽히나 타입이 link와 혼동.
- Device: status down인데 perf로 오진하는 패턴 등.

### 3.6 Diagnosis factorization audit (9.13 prep)

- 72-way Cartesian, type-only partial 없음.
- Fault-specific reward shaping은 지양 → generic protocol variant 선호.

---

## 4. 숏컷 깨기 시도 (보상·마스크·구조)

공통: Star, MaskablePPO 30k, seed 1, eval 100.  
Control은 재학습마다 Exact가 7~20%로 흔들릴 수 있음 → **같은 zip끼리만 비교**.

### 4.1 Symmetric reward (9.13-A)

- 가설: 보상 대칭이 숏컷을 줄인다.
- **결과:** Exact 19%→4%. **fail / inconclusive.**
- 리포트: `symmetric_reward_poc.md`

### 4.2 Diagnosis gate (9.13-B)

- 개입: 진단 전 diagnostic 툴 ≥1 강제.
- **결과:** shortcut 100%→0%, 그러나 **`scan → ping(고정) → diagnose` dummy**. Exact↓.
- **Verdict:** `dummy_tool_collapse`
- 리포트: `diagnosis_gate_poc.md`

### 4.3 Factorized diagnosis (9.13-C)

- 개입: type commit → location commit 분리.
- **결과:** Exact 0%, AvgSteps **15** (timeout). 제출 회피.
- **Verdict:** `diagnosis_avoidance_collapse`  
  (틀린 진단 −16 vs timeout이 덜 아픔)
- 리포트: `factorized_diagnosis_poc.md`

### 4.4 E1 Pure PBRS

- 스펙: `E1_PBRS_spec_수정.md`
- 개입: 증거 비트 Φ에 potential shaping. Eval shaping off.
- **결과:** Control과 동일 — NL-DSR 5%, shortcut 100%, AvgSteps 2.0. Acquisition 0%.
- **Decision: no_go**
- 리포트: `e1_pbrs_poc.md`

### 4.5 E2 Timeout = incorrect diagnosis

- 스펙: `E2_timeout_spec.md`
- 개입: 진단 없이 timeout 시 −16.
- **결과:** shortcut 여전히 100%, AvgSteps 2.0. Aggregate DSR 상승은 Link 숏컷.
- **Decision: no_go**
- 리포트: `e2_timeout_poc.md`

### 4.6 E3 `ent_coef=0.01`

- 스펙: `E3_exploration_spec.md`
- 개입: entropy coef만 변경.
- **결과:** train/sto에서는 이미 tool-use ~35–43%. **Greedy는 여전히 AvgSteps 2.0, shortcut 100%.**
- Precommitted 기준상 sto tool-use로 go였으나, “탐색이 열렸다”기보다 Control도 이미 sto에서 도구를 씀. Greedy 숏컷은 안 죽음.
- 리포트: `e3_exploration_poc.md`

### 4.7 Ablation-NL

- 스펙: `Ablation_NL_spec.md` / `_수정.md`
- 개입: **학습 분포에서 Link GT 제거** (보상 변경 없음).
- **결과:** NL-only eval에서 Control과 동일 — Device 0, shortcut 100%, AvgSteps 2.0.
- **Decision: no_go** — Link 경쟁 제거만으로는 greedy 숏컷이 안 흔들림.
- 리포트: `ablation_nl_poc.md`

---

## 5. 관측·증거 게이트 계열

### 5.1 Stochastic evidence audit

- 재학습 없음. E3 Control 등, NL-only sto n=400.
- 도구를 써도 `P(exact ∧ consistent | tool)` ≈ **0.7%**.
- Obs 구멍 재확인 → 래퍼 필요.
- 리포트: `stochastic_evidence_audit.md`

### 5.2 Obs wrapper → (조건부) E4

- 스펙: `Obs_then_E4_spec.md`
- **Stage A:** 유닛 테스트 1–5 통과 (DOWN/latency/discovery=2 기입).
- **Stage B:** Control-Obs 재학습 30k. Device ΔP(correct) **+0.000**, Perf **−0.002**. sto informed exact 0%.
- **Decision: no_go.** Stage C E4(diagnose-mask curriculum) **구현하지 않음**.
- 해석: 벡터에 숫자를 넣어도 30k PPO diagnose head가 안 움직임.
- 리포트: `obs_then_e4_stage_b.md`  
- 코드: `src/evidence_obs.py`, `src/run_obs_then_e4.py`

### 5.3 Evidence diagnose-gate

- 개입: obs에 fault-like 비트(DOWN / latency≥15ms / discovery=2 / failed tool) 있을 때만 진단 unlock. Healthy dummy ping으로는 unlock 안 됨.
- **결과:** shortcut 0%, timeout **~76%**, Exact 0%. 회피.
- **Decision: no_go**
- 리포트: `evidence_diagnose_gate_poc.md`

### 5.4 Repeat penalty (−2 on reused action_id)

- Evidence-gate 위에 같은 action 재사용 −2.
- **결과:** timeout 78%→19%, submit 22%→81%. Exact 낮음. Perf 일부 cons. Device/Misconfig cons 0.
- **Decision: ambiguous**
- 리포트: `repeat_penalty_poc.md`

### 5.5 Train episode log

- Control-Obs / gate / repeat 학습 JSONL.
- Control-Obs: reward −7.2→−3.5, steps↓, 사실상 숏컷 학습.
- Gate/repeat: length~7에 stuck, informed 곡선이 크게 안 오름.
- 리포트: `train_episode_log.md`

---

## 6. 모방·규칙·천장 (전환점)

### 6.1 BC-informed (GT 전문가 3-step)

- 코드: `src/scripted_expert.py`, `src/run_bc_informed.py`
- 전문가: GT로 `scan → 증거툴 1개 → exact diagnosis` (2000 demos, 40 epoch CE).
- Eval: evidence_obs ON, diagnose-gate OFF. 정책은 GT 안 봄.
- **결과:** Exact **58%**, shortcut 0%, AvgSteps 3.3  
  - Link 25/25, Misconfig 24/25, Device **2/25**, Perf **7/25**
- BC loss 끝 **0.69** (덜 수렴).
- Device/Perf 실패 원인 (평가 분해):
  - Device: 타입은 종종 맞지만 위치가 `edge_4` 등으로 **모달 고정**.
  - Perf: `center→edge_4` 모달; GT가 그 스포크일 때만 exact.
  - Device↔Perf 혼동 있음. “ping 후 전부 link로 제출”은 아님.
- 해석: **3-step 리듬**은 복제, **행/셀 → 진단 액션 바인딩**과 **겹침 해소**는 복제 실패.
- 리포트: `bc_informed_poc.md`  
- ckpt: `models/checkpoints/masked_ppo_bc_informed_star.zip`

### 6.2 Rule ceiling (GT-free)

- 코드: `src/rule_agent.py`, `src/run_rule_ceiling.py`
- **GT를 액션에 사용하지 않음.** scan discovery + tool JSON만.
- 트리 요지:
  1. `scan`
  2. Star 기대 간선 중 **결손** 있으면 → endpoints `check_status` → `check_interfaces` → **Link vs Misconfig**
  3. 결손 없으면 → spoke `ping` → 고지연이면 **Perf**; 실패면 `check_status` → **Device**
- **결과 (100 ep):** Exact **100%**, AvgSteps **5.8**, shortcut 0%, timeout 0%.  
  Device/Perf/Misconfig/Link **각 25/25**.
- 의미: Star 단일 고장은 **조건식으로 닫히는 문제**. “툴 결과 표 → 분류” 상한 = 사실상 만점.
- 리포트: `rule_ceiling_poc.md`

### 6.3 BC-rule (Rule 궤적 모방) ← 현재 best learned

- 코드: `src/run_bc_rule.py`
- 데모: RuleAgent 성공 궤적 2000 ep (~11675 transitions, avg len 5.8). **30k PPO 아님.** BC 40 epoch.
- BC loss 끝 **0.0145** (informed 0.69와 대조).
- **결과:** Exact **99%**, shortcut 0%, AvgSteps 6.0  
  - Device 25/25, Perf 24/25, Misconfig 25/25, Link 25/25  
  - hard-fault Go 규칙 기준 **go**
- 해석: 같은 MultiInputPolicy로도 **겹침 해소 트리가 데모에 있으면** Device/Perf까지 학습 가능.  
  이전 BC-informed 실패는 “MLP 불가”가 아니라 **GT 3-step 데모가 트리를 안 담음**.
- 리포트: `bc_rule_poc.md`  
- ckpt: `models/checkpoints/masked_ppo_bc_rule_star.zip`

### Rule vs BC-informed vs BC-rule

| | Rule (GT 없음) | BC-informed | BC-rule |
|---|---|---|---|
| 정체 | 조건식 | 신경망 모방 | 신경망 모방 |
| 데모 출처 | 없음 | GT 전문가 3-step | Rule 궤적 |
| 학습 | 없음 | BC 40 ep | BC 40 ep |
| 실행 시 GT | 안 봄 | 안 봄 | 안 봄 |
| Exact | **100%** | 58% | **99%** |
| Device | 25/25 | 2/25 | 25/25 |
| Perf | 25/25 | 7/25 | 24/25 |

---

## 7. 고장 미분 (왜 여러 툴이 필요한가)

Cisco `docs/fault-behaviors.md` + 벤치마크 테스트와 일치.

| 증상 | 후보 | 가르는 툴 |
|---|---|---|
| ping 실패 | Link / Device / Misconfig | `check_status`, 반대 ping, `check_interfaces` |
| 스캔상 결손 간선 | Link (양방향) / Misconfig (한쪽) | 비대칭 + interfaces |
| 스캔상 전부 연결 | Device / Perf | status DOWN vs ping 성공+고지연 |
| ping 성공 + 지연 10–20× | **Perf만** | (문서: ping이 성공하는 유일한 고장) |

한 툴 결과가 “A or B”로 보이는 경우가 정상이다. Rule/BC-rule은 그 겹침을 **두 번째 툴로 푼다.**  
GT 3-step 전문가는 겹침을 **건너뛰고** 정답 툴을 바로 고른다 → 정책이 그 조건부 분기를 못 배움.

---

## 8. Cisco 벤치마크 검증 규모 (참고)

Assessment 문서 (`docs/tutorials/running-assessments.md`):

- 기본 **`num_episodes = 100`**
- snapshot 모드면 스냅샷 개수에 맞춤
- 로컬 스모크 `local_test.toml`: 2 ep
- SB3 가이드 학습 예시: **100k timesteps** (에피소드 수 아님)

우리 PoC eval 100ep는 Cisco assessment 스케일과 같다.  
설정은 다름: 우리는 주로 Star·6 devices·15 steps; Cisco assessment 예시는 max_devices 8, steps 25, star/mesh/hierarchical.

---

## 9. 하지 말아야 할 것 / 잠긴 해석

이미 데이터로 잠근 것 (뒤집어쓰지 말 것):

1. Vanilla PPO → invalid collapse.
2. MaskablePPO → invalid 고정 + greedy `scan→diagnose` Link 숏컷.
3. Link-only는 스캔 분리 가능해서 100% 가능.
4. “Cisco PPO가 틀렸다” / “fault unlearnable” 주장 금지 — Rule·BC-rule이 반증.
5. Obs 래퍼만으로 Stage B no_go → E4만 올리지 말 것 (스펙대로 Stage B Go일 때만).
6. 공식 보상을 바꿔 놓고 Control과 같은 과제라고 부르지 말 것.
7. 같은 mixture에서 timesteps만 늘리거나 +16만 키워도 greedy argmax 숏컷이 Device/Perf 트리로 뒤집히지 않았음 (Control-Obs·gate plateau).

권장하지 않는 반복:

- 또 다른 보상만 패치한 30k MaskablePPO
- GT 3-step BC epoch만 늘리기
- Mesh를 Seen troubleshooting 없이 올리기

---

## 10. 다음에 가능한 방향

현재 Star Seen에서 **학습된** best는 BC-rule (99%).

| 방향 | 내용 | 주의 |
|---|---|---|
| A. Mesh / Unseen | BC-rule 정책 또는 Rule을 mesh에 평가 | Rule은 star 가정(center/spokes) → mesh용 트리 재작성 필요 |
| B. Strong BC → gated PPO | 풀 Rule-BC 후 gate PPO | **완료:** Exact 77% go. Misconfig 잔여. zip 크로스프로세스 로드는 pad/동일프로세스 권장 |
| C. Probe + 표 분류기 | 툴 결과를 컬럼화 후 작은 classifier | 상한/베이스라인; “PPO가 배웠다”와 다름 |
| D. LLM / AgentBeats | Cisco 주 대회 경로와 비교 | RL 논문 주장과 채널 분리 |

---

## 11. 재실행 치트시트

워크스페이스: `netheal-topology-generalization/`

```powershell
$env:PYTHONPATH="...\netheal-ai-agent-benchmark"
conda run -n netheal_base --no-capture-output python -m src.<module>
```

| 실험 | 모듈 |
|---|---|
| Rule ceiling | `src.run_rule_ceiling` |
| BC-rule | `src.run_bc_rule` |
| BC-informed | `src.run_bc_informed` |
| Obs Stage A/B | `src.run_obs_then_e4` |
| Evidence gate | `src.run_evidence_diagnose_gate` |
| BC-seed + gate PPO | `src.run_bc_seed_gate_ppo` |
| Full BC-rule + gate PPO | `src.run_bc_rule_gate_ppo` |
| Repeat penalty | `src.run_repeat_penalty_poc` |
| E1 / E2 / E3 | `src.run_e1_pbrs_poc` 등 |
| Ablation-NL | `src.run_ablation_nl_poc` |

프로토콜: `config/protocol.yaml`  
체크포인트: `models/checkpoints/`  
결과 JSON: `results/`  
리포트 MD: `reports/`

---

## 12. 관련 스펙 문서 (레포 루트)

| 파일 | 내용 |
|---|---|
| `Obs_then_E4_spec.md` | Obs 래퍼 → 조건부 E4 (Stage B no_go로 E4 폐쇄) |
| `E1_PBRS_spec_수정.md` | Pure PBRS |
| `E2_timeout_spec.md` | Timeout = wrong dx |
| `E3_exploration_spec.md` | ent_coef |
| `Ablation_NL_spec.md` / `_수정.md` | Link 제거 학습 분포 |
| `netheal_topology_generalization_research_plan.md` | 전체 연구 계획 (장문) |

---

## 13. 타임라인 (압축)

```
Vanilla PPO collapse
  → MaskablePPO: Link shortcut
  → Link-only 100% / post-scan audits
  → Symmetric / Gate(dummy) / Factorized(timeout)
  → E1 PBRS / E2 timeout / E3 entropy / Ablation-NL  (all fail to kill greedy shortcut)
  → Obs holes found → evidence_obs wrapper → Stage B no_go (E4 skipped)
  → Evidence gate (timeout) / Repeat penalty (ambiguous)
  → BC GT-3-step 58% (Device/Perf fail)
  → Rule ceiling 100%
  → BC-rule 99% go
  → Star pad12 scale: N=6 base 97%; zero-shot 76%@8 / 64%@10; naive FT forgets
```

---

## 14. Star scale + pad12 (2026-09-06)

상세: `netheal-topology-generalization/reports/star_scale_poc.md`

- 고정 공간: obs pad + canonical action map (`src/star_pad.py`), Discrete≈589.
- Base Rule-BC N=6 → Exact **97%**.
- Rule ceiling N=8/10: **97% / 99%**.
- Zero-shot: **76% / 64%** (shortcut 0%).
- Few-shot FT 300×10 without replay: Exact **하락**, N=6 retention 97%→35%.
- 교훈: 스케일 FT는 **리플레이 혼합**이 필요. 매번 풀 Rule 재학습은 불필요하나, 소량 데모만으로 덮어쓰면 안 됨.

---

## 15. Short Rule-BC seed + evidence-gate PPO (2026-09-06)

상세: `netheal-topology-generalization/reports/bc_seed_gate_ppo_poc.md`

의도: “조금 BC로 시드 → 조사 없이 진단 금지(gate) → PPO”.

| Stage | Exact | Shortcut | Timeout |
|---|---:|---:|---:|
| Seed BC 400×15 (gate OFF) | **0%** | 0% | 46% |
| + Gate PPO 30k | **2%** | 0% | 50% |
| Gate-only (prior) | ~0% | 0% | **76%** |
| Full Rule-BC (prior) | 99% | 0% | 0% |

- 짧은 seed는 Rule을 **복제하지 못함** (loss≈2.1 vs full BC ≈0.015).
- Gate는 shortcut를 계속 0%로 막음. Timeout은 gate-only보다 낮지만 Exact는 안 회복.
- 결론: **가벼운 BC seed ≠ 충분한 트리 초기화.** Gate+PPO만으로 troubleshooting으로 올라가지 않음.

---

## 16. Full Rule-BC + evidence-gate PPO (2026-09-06)

상세: `netheal-topology-generalization/reports/bc_rule_gate_ppo_poc.md`

In-process pad=6 (크로스프로세스 unpadded zip 로드 시 Exact 붕괴 관측 → canonical pad + 동일 프로세스).

| Stage | Exact | Shortcut | Timeout |
|---|---:|---:|---:|
| Full Rule-BC (gate OFF) | **98%** | 0% | 0% |
| Zero-shot under gate | 48% | 0% | 50% |
| + Gate PPO 30k | **77%** | 0% | 22% |

- Per-fault after PPO: Device/Perf/Link **100%**, Misconfig **8%** (go는 Device로 충족).
- Zero-shot gate: Link/Misconfig 0% (scan→diagnose가 증거 없이 잠김), Device/Perf는 유지.
- vs short-seed(~2%) / gate-only(~0%): **풀 BC 시드가 필요조건**.

---

*이 문서는 실험 기록용이다. 수치의 권위는 각 `reports/*_poc.md`와 `results/*.json`에 있다. Control Exact는 재학습마다 달라질 수 있으니 cross-run 비교 시 zip을 고정할 것.*
