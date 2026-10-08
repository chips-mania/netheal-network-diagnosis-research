# Step 1–9.10 Progress Summary

Date: 2026-09-06

**전체 실험 상세 정리:** 레포 루트 `NetHeal_experiment_history.md`  
**결과 취합:** `NetHeal_RL_results_summary.md`  
**다음 설계:** `GNN_expert_guided_spec.md`

## Next

**전체 취합 보고서 (RL 시작~GNN):** 레포 루트 `GNN_expert_guided_results_report.md`

**Pointer curriculum ✅ insufficient** (`reports/p1_pointer_curriculum.md`):  
Spoke@N10 top-1 **26%** (vs node_only 20%) but top-3 **33%** (vs 87%).  
Decision: `curriculum_insufficient_try_features`. Next: missing-incident node features.

---

## Latest: Graph expert BC Phase 3 — **go_scale_only** (post-audit)

See `reports/graph_expert_poc.md`

| | Star6 | Star8 | Star10 | Mesh6 |
|---|---:|---:|---:|---:|
| B1 MLP | 89% | 76% | **57%** | N/A (0) |
| P1 GNN+ptr | **100%** | **90%** | 42% | 9% |

Prior (broken obs): P1 74/70/60/14. After mem+id_rank fix: Seen/Star8 strong; N=10 & Mesh still weak.

Run: `python -m src.run_graph_expert_bc`

---

## Latest: P1 train audit ✅ tiny PASS

See `reports/p1_train_audit.md` — fixes: evidence mem, pre-choose snapshot, id_rank.

---

## Latest: Graph interface smoke ✅ **go**

See `reports/graph_interface_smoke.md`

- `build_graph_obs`: Star/Linear/Mesh node·edge tensors OK  
- `pointer_actions`: N=4/6/8 action_space 69/151/265, same ping resolves to different flat ids  

Run: `python -m src.run_graph_interface_smoke`

---

## Latest: Rule v2 ceiling ✅ **go**

See `reports/rule_v2_ceiling.md`

| Setting | Exact |
|---|---:|
| Star N=6 | **100%** |
| Star N=8 | **99%** |
| Star N=10 | **99%** |
| Linear N=6 | **100%** |
| Mesh N=6 | **100%** |

Run: `python -m src.run_rule_v2_ceiling`

---

## Latest: Full Rule-BC + gate PPO ✅ **go**

See `reports/bc_rule_gate_ppo_poc.md`

| Stage | Exact | Shortcut | Timeout |
|---|---:|---:|---:|
| Full Rule-BC (gate OFF) | **98%** | 0% | 0% |
| Zero-shot under gate | 48% | 0% | 50% |
| + Gate PPO 30k | **77%** | 0% | 22% |
| Short-seed + gate (prior) | ~2% | 0% | 50% |
| Gate-only (prior) | ~0% | 0% | **76%** |

Per-fault after PPO: Device/Perf/Link **100%**, Misconfig **8%**.

Run: `python -m src.run_bc_rule_gate_ppo`

---

## Latest: BC-seed + gate PPO — ambiguous

See `reports/bc_seed_gate_ppo_poc.md`

| | Exact | Shortcut | Timeout |
|---|---:|---:|---:|
| Seed BC (400×15, gate OFF) | **0%** | 0% | 46% |
| Seed+Gate PPO 30k | **2%** | 0% | 50% |
| Gate-only (prior) | ~0% | 0% | **76%** |
| Full Rule-BC (prior) | 99% | 0% | 0% |

Run: `python -m src.run_bc_seed_gate_ppo`

---

## Latest: Star scale + pad12 FT

See `reports/star_scale_poc.md`

| N | Rule | Zero-shot | Few-shot FT |
|---:|---:|---:|---:|
| 6 | 100% | **97%** (base) | — |
| 8 | 97% | **76%** | 66% (worse) |
| 10 | 99% | **64%** | 45% (worse) |

Run: `python -m src.run_star_scale_poc`

---

## Latest: BC on Rule trajectories ✅ **go**

See `reports/bc_rule_poc.md`

| | Rule | BC-rule | BC-informed (GT 3-step) |
|---|---:|---:|---:|
| Exact | 100% | **99%** | 58% |
| Device | 25/25 | **25/25** | 2/25 |
| Perf | 25/25 | **24/25** | 7/25 |
| Misconfig | 25/25 | **25/25** | 24/25 |
| Link | 25/25 | **25/25** | 25/25 |
| AvgSteps | 5.8 | 6.0 | 3.3 |
| BC loss | — | 0.015 | 0.69 |

Run: `python -m src.run_bc_rule`

---

## Rule ceiling ✅ **100% exact**

See `reports/rule_ceiling_poc.md`

GT를 액션에 쓰지 않음. scan 결손 간선 → status/interfaces로 Link vs Misconfig; 결손 없으면 spoke ping → Perf 또는 실패 끝 status → Device.

| Metric | Rule | BC-informed |
|---|---:|---:|
| Exact | **100%** | 58% |
| Device | 25/25 | 2/25 |
| Perf | 25/25 | 7/25 |
| Misconfig | 25/25 | 24/25 |
| Link | 25/25 | 25/25 |
| AvgSteps | 5.8 | 3.3 |

Run: `python -m src.run_rule_ceiling`

---

## Latest: Obs-wrapper Stage A/B ✅ **no_go**

See `reports/obs_then_e4_stage_b.md` and `Obs_then_E4_spec.md`

래퍼 ON, 원본 reward, MaskablePPO 30k seed 1. E1/E2/E3/Gate 미혼합. 패키지 미수정.

| | 결과 |
|---|---|
| Unit tests 1–5 | all pass (feat0 DOWN / latency / discovery=2 기입됨) |
| Device ΔP(correct) | **+0.0000** (dummy +0.0001) |
| Perf ΔP(correct) | **−0.0019** |
| sto P(ec \| Φ≥1) | **0.0%** (audit 2.4%) |
| sto P(ec \| tool) | **0.0%** (audit 0.7%) |
| Frozen E3 zip + wrapper | ΔP ≈ 0 (음수 대조, 예상대로) |

칸은 보이는데 30k PPO가 진단을 안 움직임. **Stage C E4 미구현.**

Run: `python -m src.run_obs_then_e4`

---

## Stochastic evidence / obs conditioning audit ✅

See `reports/stochastic_evidence_audit.md`

재학습 없음. E3 Control + Ablation-NL, NL-only sto **n=400**.

| | Control | Ablation-NL |
|---|---:|---:|
| P(Φ≥1 \| tool-use) | 28.5% | 25.0% |
| P(exact ∧ consistent \| tool-use) | **0.7% (1/144)** | **0.7% (1/148)** |
| P(exact ∧ consistent \| Φ≥1) | 2.4% | 2.7% |
| Device / Misconfig exact-cons | 0 / 0 | 0 / 0 |

det EC=0%는 동어반복. sto에서도 informed exact는 없다 (갈래 1 약함).

Obs: `device_status[:,0]`는 `status=='active'` 인코딩이라 DOWN이 안 들어감. latency는 `connection_properties`에만 있고 Gym `discovery_matrix`에 없음. Φ=2 ping 이후 P(correct device)≈0.000. dummy ping이 evidence보다 diagnose TV가 큼.

**E4 전에 obs 래퍼(status+latency) 재측정.** mask curriculum만 올리지 않음.

---

## Ablation-NL (seed 1) ✅

See `reports/ablation_nl_poc.md` and `Ablation_NL_spec_수정.md`

GT sampler에서 Link만 제거. Reward/72-way/E1/E2/E3 없음. Control = E3 zip을 같은 NL-only eval로 재평가.

| Metric | Control det | Ablation-NL det |
|---|---:|---:|
| Non-link DSR | 9% | 9% |
| Device | 0/34 | 0/34 |
| Perf | 9/33 (27%) | 9/33 (27%) |
| Misconfig | 0/33 | 0/33 |
| Shortcut / AvgSteps | 100% / 2.0 | 100% / 2.0 |
| Evidence Acquisition | 0% | 0% |

Control 4-fault det Exact 20% (E3 zip OK). **Decision: no_go.** Link 경쟁 제거는 greedy shortcut을 안 흔들음. seed 2/3 안 함. 다음: E4 diagnose-mask curriculum.

Run: `python -m src.run_ablation_nl_poc --skip-train --eval-episodes 100`

---

## E3 Exploration + ent_coef=0.01 (seed 1) ✅

See `reports/e3_exploration_poc.md` and `E3_exploration_spec.md`

Control 재학습(Stage A) + E3=`ent_coef=0.01` only. Reward/E1/E2/Gate 없음. Go/No-Go는 stochastic eval.

| Metric | Control sto | E3 sto | Control det | E3 det |
|---|---:|---:|---:|---:|
| AvgSteps | 2.58 | 2.74 | 2.0 | 2.0 |
| Tool-use before dx (non-link) | 35% | 43% | 0% | 0% |
| P(tool\|post-scan) | 0.35 | 0.42 | 0.00 | 0.00 |
| Shortcut | 66% | 54% | **100%** | **100%** |
| Exact DSR (ref) | 6% | 13% | 20% | 17% |

Stage A: entropy −3.40→−2.71, Control `P(tool|post-scan)` 학습 내내 0.37–0.58. **도구 미방문이 아님.** 죽은 것은 greedy mode.

**Precommitted Go/No-Go: go** (E3 sto tool-use 43% ≥ 10%). Control도 이미 35%라 `ent_coef`가 탐색을 연 것은 아님. Stage C(PBRS) 올리지 않음. 다음: diagnose-mask curriculum. seed 2/3·`ent_coef=0.05` 안 돌림.

Run: `python -m src.run_e3_exploration_poc --skip-train --eval-episodes 100`

---

## E2 Timeout-as-Incorrect-Diagnosis (seed 1) ✅

See `reports/e2_timeout_poc.md` and `E2_timeout_spec.md`

Timeout without diagnosis gets original incorrect-diagnosis penalty (−16). Exact/wrong/partial unchanged. Not mixed with E1/Symmetric.

| Metric | Control | E2 |
|---|---:|---:|
| Exact DSR | 10% | 18% |
| Non-link DSR | 5% | 8% |
| Shortcut Rate | **100%** | **100%** |
| AvgSteps | 2.0 | 2.0 |
| Link DSR | 24% | 48% |

**Go/No-Go: no_go** — non-link Δ +2.7%p, shortcut Δ 0. Seed 2/3 not run. Aggregate DSR rise is Link shortcut, not troubleshooting.

Run: `python -m src.run_e2_timeout_poc --skip-train --eval-episodes 100`

---

## E1 Pure PBRS (seed 1) ✅

See `reports/e1_pbrs_poc.md`

Unit tests 1–8 passed. α=1, γ=0.99. Eval shaping off.

| Metric | Control | E1 |
|---|---:|---:|
| Exact DSR | 7% | 7% |
| Non-link DSR | 5% | 5% |
| Evidence Acquisition (non-link) | 0% | 0% |
| Shortcut Rate | 100% | 100% |
| AvgSteps | 2.0 | 2.0 |

**Go/No-Go: no_go** — Δ non-link DSR +0%p, Δ acquisition +0%p. Both still `scan → diagnose`. Seed 2/3 not run. Dense credit did not open evidence-seeking. Next: E2 (terminal asymmetry), not 100k/Mesh.

Run: `python -m src.run_e1_pbrs_poc --skip-train --eval-episodes 100`

---

## Step 9.13-C Factorized Diagnosis PoC ✅

See `reports/factorized_diagnosis_poc.md`

| Metric | Control (Masked) | Factorized |
|---|---:|---:|
| Exact DSR | **12%** | **0%** |
| Type Acc | 12% | 0% |
| Loc Acc | 55% | 0% |
| Shortcut Rate (scan→dx) | **100%** | 0% |
| Factorized Shortcut | 0% | 0% |
| AvgSteps | 2.0 | **15.0** |
| Tools before dx | 1.0 | 15.0 (timeout) |

**Verdict: diagnosis_avoidance_collapse** — type→location split did not create a troubleshooting policy. The agent never commits type or submits location; it scans, loops `ping(edge_1→edge_3)`, and times out. Timeout (−1.5) beats a wrong diagnosis (−16.1). Hold 100k and Mesh.

Run: `python -m src.run_factorized_diagnosis_poc --skip-train --eval-episodes 100`

---

## Step 9.13-B Diagnosis Gate Ablation ✅

See `reports/diagnosis_gate_poc.md`

| Metric | Control (Masked) | Gate |
|---|---:|---:|
| Exact DSR | **19%** | **10%** |
| Type Acc | 23% | 32% |
| Loc Acc | 48% | 28% |
| Shortcut Rate | **100%** | **0%** |
| AvgSteps | 2.0 | 3.0 |
| Tools before dx | 1.0 | 2.0 |

**Verdict: dummy_tool_collapse** — Gate killed `scan→diagnose` but policy learned `scan → ping(edge_4→edge_2) → diagnose`. Link DSR 76%→28%. Next: **9.13-C Factorized Diagnosis**.

Run: `python -m src.run_diagnosis_gate_poc --skip-train --eval-episodes 100`

---

## Step 9.13-A Symmetric Reward PoC ✅

See `reports/symmetric_reward_poc.md`

| Metric | Control (Original) | Variant R (Symmetric) |
|---|---:|---:|
| Exact DSR | **19%** | **4%** |

**Verdict: inconclusive_or_fail** — reward symmetry alone insufficient.

---

See `reports/diagnosis_factorization_audit.md`

**Confirmed:**
- Diagnosis = **72 actions** = fault_type × location (Cartesian product on Star)
- Post-scan: **72 terminal diagnoses** immediately valid (+ 72 diagnostic tools)
- Reward: exact +15.9; partial **location overlap** +4.8; penalty -16.1; **no type-only partial**
- PPO marginals mirror 9.9: P(type)≈0.83 vs P(loc)≈0.09; M opposite; D both ~0

**Do not** fault-specific reward shaping next. Prefer generic protocol variants.

Run: `python -m src.run_diagnosis_factorization_audit --network-seed 42`

---

See `reports/failure_mode_audit.md`

| Fault | Failure mode | 9.11 finding |
|-------|--------------|--------------|
| M | type discrimination | scan leaves 9/10 discovery cells (blocked port ≈ missing link); PPO preserves loc in wrong-type preds |
| D | type mapping | status=down + ping fail after tools; PPO still says performance_degradation |
| P | location | check_interfaces(center) shows latency spike on faulted edge; PPO never runs it, guesses edge_1→center |

Run: `python -m src.run_failure_mode_audit --network-seed 42`

---

See `reports/tool_solvability_audit.md`

Purpose: check whether NetHeal tools expose enough evidence to identify fault type + location, **before** blaming PPO / running 100k / curriculum.

Run:
```powershell
conda run -n netheal_base --no-capture-output python -m src.run_tool_solvability_audit --network-seed 42
```

---

## Step 9.9 Non-link Single-Fault Audit ✅ (wording corrected)

See `reports/nonlink_single_fault_learnability.md`

**Correct interpretation:** under MaskablePPO / 30k / seed 1, no valid troubleshooting policy was learned for D/P/M. This does **not** prove unsolvability.

| Model | DSR | Type acc | Loc acc | Verdict |
|-------|-----|----------|---------|---------|
| D | 0% | 0% | 55% | no_valid_policy_30k |
| P | 19% | 100% | 19% | no_valid_policy_30k (location guess) |
| M | 22% | 22% | **100%** | no_valid_policy_30k (type confusion) |

**Misconfiguration:** location accuracy 100% at episode level — aggregate count overlap was misleading; PPO reads location but not fault type.

**Next:** use 9.10 solvability table to decide RL vs environment vs task-definition issue per fault.

---

## Step 9.8 Post-Scan Information Audit ✅

See `reports/post_scan_information_audit.md`

### Controlled comparison (seed=42, anchor edge_0 / center→edge_0)

| Fault | Post-scan discovery nonzero | Separable from others? |
|-------|----------------------------|-------------------------|
| link_failure | **8** (missing edge) | **Yes** |
| device_failure | 10 | No (same as perf/misconfig) |
| performance_degradation | 10 | No |
| misconfiguration | 10 | No |

### Scan-only fault-type probe (200 ep, CV)

| Model | Accuracy | Random |
|-------|----------|--------|
| Logistic / DT / RF | **50%** | 25% |

→ **50% = link vs non-link only** (3 fault types identical post-scan)

### Implications

1. Link-only PPO 100%: scan exposes **missing link** → location diagnosable without ping.
2. Multi-fault link collapse: strongest scan-only signal is link pattern.
3. Device/perf/misconfig need **additional tools** (ping/status/interface) — original troubleshooting framing still valid for 3/4 fault types.
4. Research pivot: **partial** — not full “scan solves everything”; split metrics by fault type.

### 9.9 + 9.10 synthesis

- **9.9:** PPO did not learn valid policies at 30k (not proof of unsolvability).
- **9.10:** All three non-link faults have **sufficient** tool evidence vs healthy; fault types separable at anchor; locations separable across edge_0/1/2.
- **Conclusion:** Environment exposes diagnosable evidence → failures are **RL / optimization / reward** issues, not benchmark-level “no information”.
- **Still hold 100k/curriculum** until per-fault failure mode is understood (e.g. M: loc 100% / type 22%; D: tools used but wrong type).

### Next (hold multi-fault 100k and curriculum until per-fault diagnosis)

## Commands

```powershell
python -m src.run_post_scan_audit --network-seed 42 --probe-seeds 200
```
