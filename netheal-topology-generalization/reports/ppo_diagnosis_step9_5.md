# Step 9.5 — PPO Behavior Diagnosis

Eval episodes: 100 (Star, same seeds as Step 9)

## 1. Agent Comparison

| Agent | DSR | AvgSteps | Coverage | First-step Diagnosis | Invalid Action Rate |
|---|---:|---:|---:|---:|---:|
| Random-Valid | 4.0% | 15.00 | 0.809 | 0.0% | 0.0% |
| Immediate-Random-Diagnosis | 4.0% | 1.00 | 0.000 | 100.0% | 100.0% |
| Majority-Diagnosis (link_failure) | 2.0% | 1.00 | 0.000 | 100.0% | 100.0% |
| PPO (30k sanity) | 4.0% | 1.00 | 0.000 | 100.0% | 100.0% |

> **Invalid action rate** = fraction of steps where action ∉ `get_valid_actions()`.
> The env still executes these actions (including undiscovered diagnosis), so PPO can
> terminate in 1 step with Coverage=0 despite `is_valid=False`.

## 2. PPO Action Behavior

- First action is diagnosis: **100.0%**
- Avg tool actions before diagnosis: **0.00**
- Invalid action rate: **100.0%**
- scan usage: 0.0%
- ping usage: 0.0%
- traceroute usage: 0.0%
- check_status usage: 0.0%
- check_interfaces usage: 0.0%

### Predicted diagnosis distribution (PPO)

- performance_degradation @ `edge_0->center` (action_id=102): 100/100 (100.0%)

### Actual fault distribution

- link_failure: 25/100
- device_failure: 25/100
- performance_degradation: 25/100
- misconfiguration: 25/100

### Fault-type DSR (PPO)

| Fault | DSR |
|---|---:|
| link_failure | 0.0% |
| device_failure | 0.0% |
| performance_degradation | 16.0% |
| misconfiguration | 0.0% |

### Confusion matrix (actual → predicted)

| actual \ predicted | performance_degradation |
| --- | --- |
| device_failure | 25 |
| link_failure | 25 |
| misconfiguration | 25 |
| performance_degradation | 25 |

## 3. Reward Structure

- step_penalty: -0.1
- correct diagnosis (scaled, n=6): 16.00
- wrong diagnosis (scaled, n=6): -16.00
- invalid action penalty: -5.0
- diagnosis terminates episode: True

**Scenario totals (network_size=6):**

- immediate_correct: 15.90
- immediate_wrong: -16.10
- explore_5_steps_then_correct: 15.50
- explore_10_steps_then_correct: 15.00
- explore_14_steps_then_correct: 14.60

## 4. Observation Changes After Tool Actions

- Tools change observation meaningfully: **True**
- obs0_reset → obs_after_scan_network: discovery Δ=9, device_status Δ=0, diagnostics Δ=6
- obs_after_scan_network → obs_after_ping: discovery Δ=0, device_status Δ=4, diagnostics Δ=6
- obs_after_ping → obs_after_check_interfaces: discovery Δ=0, device_status Δ=1, diagnostics Δ=6

## 5. Updated Go/No-Go Assessment

- ① Performance (PPO >> Random & Immediate): **FAIL**
- ② Behavior (troubleshooting sequence): **FAIL**
- ③ Not blind-guess-only: **FAIL**
- **Overall Step 9.5 GO: NO**
- **Verdict:** PPO ≡ collapsed immediate diagnosis (DSR 4.0% = Immediate 4.0%). Not troubleshooting.
- **100k training:** **hold** until action masking or reward/action-space fix.

## Artifacts

- `results/ppo_action_trace.csv`
- `results/ppo_diagnosis_baselines.json`
- `reports/reward_structure.md`
- `reports/observation_diff.json`

## Next step

- If NO-GO: investigate action-space masking / reward before 100k or Main Train.
- If GO: Star-only checkpoint curve (10k/30k/50k/100k) before Step 10.
