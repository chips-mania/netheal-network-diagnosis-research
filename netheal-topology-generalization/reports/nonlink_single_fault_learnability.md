# Step 9.9 — Non-link Single-Fault Learnability Audit

Config: Star, 6 devices, Hint OFF, MaskablePPO, max_steps=15, 30k steps, seed 1

Eval episodes per model: 100

> **Interpretation note:** Results show whether PPO learned a valid troubleshooting policy under this specific setup. They do **not** by themselves prove that a fault is unsolvable in NetHeal.

## Reference (Step 9.7 Link-only)

- Exact diagnosis (DSR): 100%, AvgSteps: 2.0, behavior: scan → diagnose (scan-only observability)

## Summary table

| Model | Fault | DSR | Type acc | Loc acc | AvgSteps | Tools before dx | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| D | device_failure | 0% | 0% | 55% | 3.4 | 2.4 | **no_valid_policy_30k** |
| P | performance_degradation | 19% | 100% | 19% | 2.0 | 1.0 | **no_valid_policy_30k** |
| M | misconfiguration | 22% | 22% | 100% | 2.0 | 1.0 | **no_valid_policy_30k** |

## Model D — device_failure

### Diagnosis decomposition

- Fault-type accuracy: 0%
- Location accuracy: 55%
- Exact diagnosis (DSR): 0%
- DSR margin vs Random: -1.0%

- First-step diagnosis: 0%
- scan: 100%, ping: 100%, traceroute: 15%, status: 0%, interfaces: 29%
- Action types (non-diagnosis): {'scan_network': 100, 'ping': 100, 'check_interfaces': 29, 'traceroute': 15}

### Episode-level matching (first 20)

| Ep | Actual loc | Pred type | Pred loc | Type OK | Loc OK | Exact |
|---:|---|---|---|:---:|:---:|:---:|
| 0 | `edge_2` | performance_degradation | `edge_4->edge_0` | N | N | N |
| 1 | `center` | performance_degradation | `center->edge_0` | N | Y | N |
| 2 | `edge_0` | performance_degradation | `edge_4->edge_0` | N | Y | N |
| 3 | `edge_1` | performance_degradation | `edge_4->edge_0` | N | N | N |
| 4 | `center` | performance_degradation | `center->edge_0` | N | Y | N |
| 5 | `edge_4` | performance_degradation | `edge_4->edge_0` | N | Y | N |
| 6 | `edge_3` | performance_degradation | `edge_4->edge_0` | N | N | N |
| 7 | `edge_1` | performance_degradation | `edge_4->edge_0` | N | N | N |
| 8 | `edge_4` | performance_degradation | `edge_4->edge_0` | N | Y | N |
| 9 | `edge_4` | performance_degradation | `edge_4->edge_0` | N | Y | N |
| 10 | `center` | performance_degradation | `center->edge_0` | N | Y | N |
| 11 | `edge_2` | performance_degradation | `edge_4->edge_0` | N | N | N |
| 12 | `edge_4` | performance_degradation | `edge_4->edge_0` | N | Y | N |
| 13 | `edge_4` | performance_degradation | `edge_4->edge_0` | N | Y | N |
| 14 | `edge_0` | performance_degradation | `edge_4->edge_0` | N | Y | N |
| 15 | `edge_1` | performance_degradation | `edge_4->edge_0` | N | N | N |
| 16 | `edge_1` | performance_degradation | `edge_4->edge_0` | N | N | N |
| 17 | `edge_3` | performance_degradation | `edge_4->edge_0` | N | N | N |
| 18 | `edge_4` | performance_degradation | `edge_4->edge_0` | N | Y | N |
| 19 | `edge_0` | performance_degradation | `edge_4->edge_0` | N | Y | N |
| … | (80 more episodes) | | | | | |

## Model P — performance_degradation

### Diagnosis decomposition

- Fault-type accuracy: 100%
- Location accuracy: 19%
- Exact diagnosis (DSR): 19%
- DSR margin vs Random: 18.0%

- First-step diagnosis: 0%
- scan: 100%, ping: 0%, traceroute: 0%, status: 0%, interfaces: 0%
- Action types (non-diagnosis): {'scan_network': 100}

### Episode-level matching (first 20)

| Ep | Actual loc | Pred type | Pred loc | Type OK | Loc OK | Exact |
|---:|---|---|---|:---:|:---:|:---:|
| 0 | `edge_1->center` | performance_degradation | `edge_1->center` | Y | Y | Y |
| 1 | `center->edge_1` | performance_degradation | `edge_1->center` | Y | Y | Y |
| 2 | `center->edge_2` | performance_degradation | `edge_1->center` | Y | N | N |
| 3 | `edge_0->center` | performance_degradation | `edge_1->center` | Y | N | N |
| 4 | `center->edge_0` | performance_degradation | `edge_1->center` | Y | N | N |
| 5 | `center->edge_1` | performance_degradation | `edge_1->center` | Y | Y | Y |
| 6 | `edge_4->center` | performance_degradation | `edge_1->center` | Y | N | N |
| 7 | `center->edge_4` | performance_degradation | `edge_1->center` | Y | N | N |
| 8 | `edge_4->center` | performance_degradation | `edge_1->center` | Y | N | N |
| 9 | `edge_1->center` | performance_degradation | `edge_1->center` | Y | Y | Y |
| 10 | `center->edge_1` | performance_degradation | `edge_1->center` | Y | Y | Y |
| 11 | `edge_1->center` | performance_degradation | `edge_1->center` | Y | Y | Y |
| 12 | `center->edge_4` | performance_degradation | `edge_1->center` | Y | N | N |
| 13 | `edge_1->center` | performance_degradation | `edge_1->center` | Y | Y | Y |
| 14 | `center->edge_3` | performance_degradation | `edge_1->center` | Y | N | N |
| 15 | `center->edge_4` | performance_degradation | `edge_1->center` | Y | N | N |
| 16 | `edge_0->center` | performance_degradation | `edge_1->center` | Y | N | N |
| 17 | `edge_3->center` | performance_degradation | `edge_1->center` | Y | N | N |
| 18 | `center->edge_3` | performance_degradation | `edge_1->center` | Y | N | N |
| 19 | `center->edge_3` | performance_degradation | `edge_1->center` | Y | N | N |
| … | (80 more episodes) | | | | | |

### Representative successful traces

```text
Episode 0:
  Step 1 → scan_network (reward=-0.1)
  Step 2 → performance_degradation @ edge_1->center (reward=15.9)
```
```text
Episode 1:
  Step 1 → scan_network (reward=-0.1)
  Step 2 → performance_degradation @ edge_1->center (reward=15.9)
```
```text
Episode 5:
  Step 1 → scan_network (reward=-0.1)
  Step 2 → performance_degradation @ edge_1->center (reward=15.9)
```

## Model M — misconfiguration

### Diagnosis decomposition

- Fault-type accuracy: 22%
- Location accuracy: 100%
- Exact diagnosis (DSR): 22%
- DSR margin vs Random: 20.0%

- First-step diagnosis: 0%
- scan: 100%, ping: 0%, traceroute: 0%, status: 0%, interfaces: 0%
- Action types (non-diagnosis): {'scan_network': 100}

### Episode-level matching (first 20)

| Ep | Actual loc | Pred type | Pred loc | Type OK | Loc OK | Exact |
|---:|---|---|---|:---:|:---:|:---:|
| 0 | `edge_2` | misconfiguration | `edge_2` | Y | Y | Y |
| 1 | `center` | link_failure | `edge_1->center` | N | Y | N |
| 2 | `edge_0` | performance_degradation | `edge_0->center` | N | Y | N |
| 3 | `edge_1` | link_failure | `edge_1->center` | N | Y | N |
| 4 | `center` | link_failure | `edge_1->center` | N | Y | N |
| 5 | `edge_4` | performance_degradation | `edge_4->center` | N | Y | N |
| 6 | `edge_3` | misconfiguration | `edge_3` | Y | Y | Y |
| 7 | `edge_1` | link_failure | `edge_1->center` | N | Y | N |
| 8 | `edge_4` | performance_degradation | `edge_4->center` | N | Y | N |
| 9 | `edge_4` | performance_degradation | `edge_4->center` | N | Y | N |
| 10 | `center` | link_failure | `edge_1->center` | N | Y | N |
| 11 | `edge_2` | misconfiguration | `edge_2` | Y | Y | Y |
| 12 | `edge_4` | performance_degradation | `edge_4->center` | N | Y | N |
| 13 | `edge_4` | performance_degradation | `edge_4->center` | N | Y | N |
| 14 | `edge_0` | performance_degradation | `edge_0->center` | N | Y | N |
| 15 | `edge_1` | link_failure | `edge_1->center` | N | Y | N |
| 16 | `edge_1` | link_failure | `edge_1->center` | N | Y | N |
| 17 | `edge_3` | misconfiguration | `edge_3` | Y | Y | Y |
| 18 | `edge_4` | performance_degradation | `edge_4->center` | N | Y | N |
| 19 | `edge_0` | performance_degradation | `edge_0->center` | N | Y | N |
| … | (80 more episodes) | | | | | |

### Misconfiguration note

- Episodes with location correct but wrong type: 78
- Compare aggregate location counts vs episode table before inferring observability.

### Representative successful traces

```text
Episode 0:
  Step 1 → scan_network (reward=-0.1)
  Step 2 → misconfiguration @ edge_2 (reward=15.9)
```
```text
Episode 6:
  Step 1 → scan_network (reward=-0.1)
  Step 2 → misconfiguration @ edge_3 (reward=15.9)
```
```text
Episode 11:
  Step 1 → scan_network (reward=-0.1)
  Step 2 → misconfiguration @ edge_2 (reward=15.9)
```

## Overall verdict

- **no_valid_policy_under_current_settings**
- Under current settings (MaskablePPO, 30k, seed 1), no valid troubleshooting policy was learned for the three non-link faults. This does NOT yet prove the faults are unsolvable — only that PPO did not learn effective policies in this run.
- Next: Do not run multi-fault 100k yet. Run Step 9.10 solvability audit to separate environment evidence limits from RL learning limits.

### Decision guide

- **valid_policy_all_three**: valid troubleshooting per fault → masked multi-fault 100k justified.
- **mixed_policy_signal**: solvability audit on failing faults before joint training.
- **no_valid_policy_under_current_settings**: run Step 9.10 before changing reward/curriculum/100k.
