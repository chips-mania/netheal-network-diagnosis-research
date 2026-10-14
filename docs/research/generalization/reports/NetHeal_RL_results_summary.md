# NetHeal RL 실험 결과 취합

**작성일:** 2026-09-06  
**코드:** `netheal-topology-generalization/`  
**환경(수정 금지):** `netheal-ai-agent-benchmark/`  
**Conda:** `netheal_base`  
**상세 로그:** `NetHeal_experiment_history.md`  
**개별 PoC:** `netheal-topology-generalization/reports/`

이 문서는 지금까지의 PoC·감사를 **한 파일로 취합**한 결과 요약이다.  
수치의 1차 권위는 각 `reports/*_poc.md` / `results/*.json`에 있다.

---

## 0. 연구 질문과 판정

### 질문

> NetHeal의 **공식 RL 인터페이스**(Gymnasium + 원본 보상 + 진단 액션) 위에서,  
> **어떤 학습 프로토콜**이 실제 troubleshooting 정책(도구로 증거 수집 → 타입×위치 진단)을 만드는가?

Cisco 주 대회 경로(AgentBeats/LLM)와 분리. Mesh 일반화는 Seen에서 유효 정책이 생긴 뒤로 미룸.

### 한 줄 판정

| 프레임 | 판정 |
|---|---|
| 공식 보상 + MaskablePPO가 troubleshooting을 학습하는가? | **아니오** (Link 숏컷) |
| 과제가 본질적으로 풀 수 없는가? | **아니오** (Rule 100%, BC-rule 99%) |
| 병목은? | **데모/프로토콜이 겹침 해소 트리를 담느냐** (MLP 용량·72-way 불가 아님) |
| “학습된 크기/위상 불변 방법론”인가? | **아니오에 가깝다** (경로 클론 + 스케일 하락) |
| RL 변형(게이트·PBRS·보상 패치)로 발견 엔진을 만들었는가? | **아니오** (부정 결과가 본체) |

**취합 해석:** 가치는 “PPO를 고쳤다”가 아니라 **이 인터페이스의 실패 모드와 상한·병목을 닫은 감사**에 있다.

---

## 1. 고정 실험 설정 (대부분 PoC)

| 항목 | 값 |
|---|---|
| Topology | Star, 보통 **6 devices** |
| Fault types | link / device / performance / misconfiguration (균등) |
| Algo | MaskablePPO, `MultiInputPolicy` (예외 명시) |
| RL timesteps | 보통 **30k** |
| Seed | 주로 **1** |
| Eval | **100 episodes**, deterministic(greedy) 기본 |
| Max steps | 15 |
| Cisco 패키지 | **미수정** — 래퍼/프로토콜만 실험 레포 |

### 공식 보상 (요지)

| 이벤트 | 대략값 (6 devices) |
|---|---|
| Exact 성공 | **+16** (도구 사용 여부와 **무관**) |
| Wrong 진단 | **−16** |
| Type-only partial | **없음** |
| Step | **−0.1** |

진단 공간 ≈ **72-way** (type × location). Post-scan 직후에도 진단이 즉시 valid.

### 관측 구멍 (요약)

시뮬레이터 툴 결과에는 증거가 있으나 Gym `to_dict()`가 DOWN/latency 등을 잘 안 노출.  
실험 래퍼 `evidence_obs`가 같은 키·shape에 채움. **새 obs 키 추가 없음.**

---

## 2. 결과 총표

| 단계 | Exact / 핵심 지표 | Decision | 한 줄 |
|---|---|---|---|
| Vanilla PPO | invalid collapse | fail | 무효 진단 붕괴 |
| MaskablePPO 30k | DSR~17%, AvgSteps **2.0**, shortcut성 | no_go | `scan→diagnose` Link 숏컷 |
| Link-only 30k | DSR **~100%** | — | 스캔만으로 링크 위치 가능 |
| Symmetric reward | Exact↓ | fail | 숏컷 미해결 |
| Diagnosis gate (≥1 tool) | shortcut 0%, dummy ping | dummy_tool | 형식만 도구 |
| Factorized diagnosis | Exact 0%, timeout | avoidance | 제출 회피 |
| E1 Pure PBRS | shortcut 100%, AvgSteps 2.0 | **no_go** | Control과 동일 |
| E2 Timeout=−16 | shortcut 100% | **no_go** | greedy 숏컷 유지 |
| E3 ent_coef=0.01 | greedy AvgSteps 2.0 | (sto 형식 go) | greedy 숏컷 안 죽음 |
| Ablation-NL (학습에서 Link 제거) | NL eval shortcut 100% | **no_go** | Link 경쟁 제거로 안 풀림 |
| Obs wrapper Stage B | Device ΔP(correct)≈0 | **no_go** | E4 미실시 |
| Evidence diagnose-gate | Exact~0%, timeout **~76%** | **no_go** | 회피 |
| Repeat penalty (−2) | submit↑, Exact 낮음 | **ambiguous** | Device/Misconfig 약함 |
| BC-informed (GT 3-step) | Exact **58%** | — | Device/Perf 붕괴 |
| Rule ceiling (GT-free) | Exact **100%** | ceiling | 조건식 트리로 닫힘 |
| **BC-rule** | Exact **99%** | **go** | best learned (모방) |
| Star pad12 N=6 base | Exact **97%** | — | 고정 공간 BC |
| Zero-shot N=8 / N=10 | **76% / 64%** | — | 크기↑ 성능↓ |
| Naive FT (no replay) | N=8/10 **악화**, N=6 망각 | — | forgetting |
| Replay FT | N=6 유지, held-out≈zero-shot | — | 망각만 방지 |
| Mix {4,5,6}→N=8 | held-out **75%** ≈ N6-only | — | 일반화 이득 없음 |
| Short BC-seed + gate PPO | Exact **2%** | **ambiguous** | seed 미모방 |
| Full BC-rule + gate PPO | Exact **77%** | **go** | 시드가 본체; Misconfig 8% |

---

## 3. 핵심 비교표

### 3.1 Rule vs BC-informed vs BC-rule

| | Rule (GT 없음) | BC-informed | BC-rule |
|---|---:|---:|---:|
| Exact | **100%** | 58% | **99%** |
| Device | 25/25 | 2/25 | 25/25 |
| Perf | 25/25 | 7/25 | 24/25 |
| Misconfig | 25/25 | 24/25 | 25/25 |
| Link | 25/25 | 25/25 | 25/25 |
| AvgSteps | 5.8 | 3.3 | 6.0 |
| BC loss (끝) | — | ~0.69 | ~0.015 |
| 데모 | 없음 (조건식) | GT `scan→툴1→정답` | Rule 궤적 |

**해석:** 같은 MLP로도 트리를 배울 수 있다. GT 3-step은 겹침 해소를 건너뛰어 Device/Perf 바인딩이 무너진다.

### 3.2 Gate / seed / PPO

| Stage | Exact | Shortcut | Timeout |
|---|---:|---:|---:|
| Gate-only PPO | ~0% | 0% | **76%** |
| Short seed (400×15) BC | **0%** | 0% | 46% |
| Short seed + Gate PPO | **2%** | 0% | 50% |
| Full Rule-BC (gate OFF, pad6) | **98%** | 0% | 0% |
| Full BC zero-shot under gate | 48% | 0% | 50% |
| Full BC + Gate PPO 30k | **77%** | 0% | 22% |

Full BC+gate PPO per-fault: Device/Perf/Link **100%**, Misconfig **8%**.

**해석:** 게이트는 숏컷을 막지만 시드 없이는 정책이 안 생긴다. 풀 BC가 있으면 PPO가 일부 회복하나 BC 단독(99%)보다 낮다.

### 3.3 Star 크기 스케일 (pad12)

| N | Rule | Zero-shot (N=6 BC) | Naive FT | Replay FT (요지) |
|---:|---:|---:|---:|---|
| 4 | 100% | ~0–22% | — | FT→4: **77%** (N6 유지) |
| 6 | 100% | **97–99%** | 망각 시 35%대 | retention **~98%** |
| 8 | ~97–99% | **76–79%** | **66%** (악화) | ~72% (≈zero-shot) |
| 10 | ~98–99% | **54–64%** | **45%** (악화) | ~53% |

Mix {4,5,6} → held-out N=8 Exact **75%** ≈ N=6-only **76–79%**.  
→ 다중 크기 데모가 “크기 불변 방법”을 만들지 못함. **경로 클론** 해석 유지.

---

## 4. 단계별 요약

### 4.1 초기 감사

- Vanilla → invalid collapse; Maskable → Link 숏컷.
- Link-only 100%: post-scan missing edge로 위치 분리.
- Non-link 단독 학습도 “유효 troubleshooting GO” 실패였으나, 이후 Rule 100%로 **unsolvable 아님** 확정.
- 툴 결과만 보면 타입+위치 구분 가능. PPO는 안 쓰거나 못 맞춤.

### 4.2 숏컷 깨기 (보상·구조) — 대체로 실패

Symmetric / Gate(dummy) / Factorized(timeout) / E1 / E2 / E3 / Ablation-NL.  
공통: **greedy `scan→diagnose`를 Device/Perf 트리로 뒤집지 못함.**

### 4.3 관측·게이트

- Stochastic audit: 도구를 써도 `P(exact∧consistent|tool)` ≈ 0.7%.
- Obs Stage B **no_go** → E4 curriculum **미실시**.
- Evidence gate: shortcut 0%, timeout ~76% (**회피**).
- Repeat penalty: 제출은 늘지만 hard-fault 미흡 (**ambiguous**).

### 4.4 전환점: Rule / BC

- Rule ceiling **100%** (GT를 액션에 안 씀).
- BC-rule **99% go** = 현재 best *learned* 정책 (모방).
- BC-informed **58%** = 데모 품질 문제의 대조군.

### 4.5 스케일·시드·게이트 PPO

- pad12로 N 전이 실험: 위로 갈수록 Exact↓; FT는 리플레이 없으면 망각.
- Short BC seed + gate: **실패** (Exact ~2%).
- Full BC + gate: **go 77%**, 그러나 Misconfig 약하고 BC보다 열등 → **시드가 본체**.

---

## 5. 잠긴 해석 (뒤집어쓰지 말 것)

1. Vanilla PPO → invalid collapse.  
2. MaskablePPO + 원본 보상 → greedy Link 숏컷.  
3. Link-only는 스캔 분리로 100% 가능.  
4. “fault unlearnable” / “72-way라서 불가” **금지** — Rule·BC-rule이 반증.  
5. Obs 채움만으로 Stage B no_go → E4만 올리지 말 것.  
6. 보상/마스크를 바꾼 실험은 **Cisco Control과 다른 named variant**.  
7. timesteps·entropy·Link 제거만으로 greedy 숏컷이 Device/Perf 트리로 안 바뀜.  
8. 현재 best learned는 **Rule 궤적 BC**; RL fine-tune/gate는 보조·제약이지 발견 엔진이 아님.  
9. 크기 스케일 결과는 **경로 클론**에 가깝고, Mesh 자격은 아직 없음.

---

## 6. 고장 미분 (왜 여러 툴이 필요한가)

| 증상 | 후보 | 가르는 툴 |
|---|---|---|
| ping 실패 | Link / Device / Misconfig | status, 반대 ping, interfaces |
| 스캔상 결손 간선 | Link / Misconfig | 비대칭 + interfaces |
| 스캔상 전부 연결 | Device / Perf | status DOWN vs ping 성공+고지연 |
| ping 성공 + 지연 10–20× | **Perf만** | (문서상 ping 성공하는 유일한 고장) |

한 툴이 “A or B”로 보이는 것이 정상. Rule/BC-rule은 **두 번째 툴**로 푼다. GT 3-step은 그 분기를 건너뛴다.

---

## 7. 산출물 인덱스

### 주요 리포트

| 주제 | 파일 |
|---|---|
| 진행 요약 | `netheal-topology-generalization/reports/step_progress.md` |
| 전체 실험 로그 | `NetHeal_experiment_history.md` |
| Rule ceiling | `reports/rule_ceiling_poc.md` |
| BC-rule / BC-informed | `reports/bc_rule_poc.md`, `bc_informed_poc.md` |
| Obs Stage B | `reports/obs_then_e4_stage_b.md` |
| Evidence gate / repeat | `reports/evidence_diagnose_gate_poc.md`, `repeat_penalty_poc.md` |
| E1–E3 / Ablation-NL | `reports/e1_pbrs_poc.md` 등 |
| Star scale / replay / mix | `reports/star_scale_poc.md`, `star_scale_replay_poc.md`, `star_mix456_poc.md` |
| Short / Full BC+gate | `reports/bc_seed_gate_ppo_poc.md`, `bc_rule_gate_ppo_poc.md` |

### 재실행 (요지)

```powershell
cd netheal-topology-generalization
$env:PYTHONPATH="...\netheal-ai-agent-benchmark"
conda run -n netheal_base --no-capture-output python -m src.<module>
```

| 실험 | 모듈 |
|---|---|
| Rule ceiling | `src.run_rule_ceiling` |
| BC-rule | `src.run_bc_rule` |
| Full BC + gate PPO | `src.run_bc_rule_gate_ppo` |
| Short seed + gate | `src.run_bc_seed_gate_ppo` |
| Star scale | `src.run_star_scale_poc` |

프로토콜: `config/protocol.yaml` · ckpt: `models/checkpoints/` · JSON: `results/`

### 스펙 (레포 루트)

`Obs_then_E4_spec.md`, `E1_PBRS_spec_수정.md`, `E2_timeout_spec.md`, `E3_exploration_spec.md`, `Ablation_NL_spec.md`, `netheal_topology_generalization_research_plan.md`

---

## 8. 타임라인 (압축)

```
Vanilla PPO collapse
  → MaskablePPO: Link shortcut
  → Link-only 100% / post-scan audits
  → Symmetric / Gate(dummy) / Factorized(timeout)
  → E1 PBRS / E2 timeout / E3 entropy / Ablation-NL  (greedy shortcut 미해결)
  → Obs holes → evidence_obs → Stage B no_go (E4 skipped)
  → Evidence gate (timeout) / Repeat penalty (ambiguous)
  → BC GT-3-step 58%
  → Rule ceiling 100%
  → BC-rule 99% go          ← best learned
  → Star pad12 scale / replay / mix4-5-6 (경로 클론)
  → Short BC-seed+gate ~2% fail
  → Full BC+gate PPO 77% go (시드 의존, Misconfig 약함)
```

---

## 9. 권고 (취합 시점)

1. **추가 RL 변형 PoC(보상·게이트·30k 재탕)는 한계효용이 바닥** — 여기서 끊는 것이 합리적.  
2. 남길 스토리는 Exact 표보다 **실패 모드 감사 + 상한(Rule) + 병목(데모 트리)**.  
3. Mesh/Unseen은 “Seen에서 RL이 방법을 학습했다”고 가정하고 가면 과대주장.  
4. **다음 연구 단계:** **GNN + Expert-guided** — `GNN_expert_guided_spec.md`  
   - **Phase 1 ✅ go:** Rule v2 ceiling  
   - **Phase 2 ✅ go:** graph_obs + pointer  
   - **Phase 3 ✅ ran — no_go_seen:** B1 vs P1 (`reports/graph_expert_poc.md`) — P1 Seen 74%·underfit; scale/mesh go 미달  

---

*이 문서는 취합본이다. Control Exact는 재학습마다 흔들릴 수 있으니 cross-run 비교 시 체크포인트 zip을 고정할 것. Named variant를 Cisco Control과 동일 과제로 부르지 말 것.*
