# Step 9.12 — Diagnosis Action / Reward Factorization Audit

Purpose: test whether failures are driven by **type×location joint diagnosis** and
**asymmetric sparse reward**, before any fault-specific reward shaping.

## 9.12-A — Diagnosis action structure

- Devices: 6 → `['center', 'edge_0', 'edge_1', 'edge_2', 'edge_3', 'edge_4']`
- Total action space: **151**
- Diagnosis actions: **72** (expected fault×location product: 72)
- Is pure type×location Cartesian product: **True**

### Diagnosis count by fault type

- device_failure: 6
- misconfiguration: 6
- link_failure: 30
- performance_degradation: 30

- Device-level locations: 6
- Link-level locations: 30

Full mapping: `results/diagnosis_action_mapping.json`

## 9.12-B — Post-scan valid actions (masking, leakage-free)

| Phase | Total valid | topology | diagnostic | diagnosis |
|---|---:|---:|---:|---:|
| pre-scan | 1 | 1 | 0 | 0 |
| post-scan | 151 | 7 | 72 | **72** |

- Post-scan **72** terminal diagnosis candidates immediately available.
- By type: {'device_failure': 6, 'misconfiguration': 6, 'link_failure': 30, 'performance_degradation': 30}

> Scan opens the full diagnosis lottery while tools remain optional — PPO can skip evidence gathering.

## 9.12-C — Reward factorization (4-way grid @ anchor)

Scaling multiplier (network_size=6): **1.6**

### link_failure @ `center->edge_0`

| Case | Type OK | Loc OK | Exact | Total reward | diagnosis | partial_device |
|---|:---:|:---:|:---:|---:|---:|---:|
| type_ok_loc_ok | Y | Y | Y | 15.90 | 16.00 | 0.00 |
| type_ok_loc_wrong | Y | N | N | 2.30 | 0.00 | 2.40 |
| type_wrong_loc_ok | N | Y | N | 4.70 | 0.00 | 4.80 |
| both_wrong | N | N | N | -16.10 | -16.00 | 0.00 |

### device_failure @ `edge_0`

| Case | Type OK | Loc OK | Exact | Total reward | diagnosis | partial_device |
|---|:---:|:---:|:---:|---:|---:|---:|
| type_ok_loc_ok | Y | Y | Y | 15.90 | 16.00 | 0.00 |
| type_ok_loc_wrong | Y | N | N | -16.10 | -16.00 | 0.00 |
| type_wrong_loc_ok | N | N | N | 4.70 | 0.00 | 4.80 |
| both_wrong | N | N | N | -16.10 | -16.00 | 0.00 |

### performance_degradation @ `center->edge_0`

| Case | Type OK | Loc OK | Exact | Total reward | diagnosis | partial_device |
|---|:---:|:---:|:---:|---:|---:|---:|
| type_ok_loc_ok | Y | Y | Y | 15.90 | 16.00 | 0.00 |
| type_ok_loc_wrong | Y | N | N | 2.30 | 0.00 | 2.40 |
| type_wrong_loc_ok | N | Y | N | 4.70 | 0.00 | 4.80 |
| both_wrong | N | N | N | -16.10 | -16.00 | 0.00 |

### misconfiguration @ `edge_0`

| Case | Type OK | Loc OK | Exact | Total reward | diagnosis | partial_device |
|---|:---:|:---:|:---:|---:|---:|---:|
| type_ok_loc_ok | Y | Y | Y | 15.90 | 16.00 | 0.00 |
| type_ok_loc_wrong | Y | N | N | -16.10 | -16.00 | 0.00 |
| type_wrong_loc_ok | N | N | N | 4.70 | 0.00 | 4.80 |
| both_wrong | N | N | N | -16.10 | -16.00 | 0.00 |

### Reward asymmetry (key finding)

- **Exact match** (type AND location): large positive reward (~+15.9 at 6 devices).
- **Wrong type + correct location**: often **partial_device_reward** (location overlap), not full penalty.
- **Correct type + wrong location**: partial_device_reward if device overlap exists.
- **No type-only partial reward** — type correctness alone is never rewarded without location overlap.
- This asymmetry can produce **complementary failures**: learn location OR type shortcut separately.

## 9.12-D — PPO diagnosis probability marginals (post-scan)

### misconfiguration

- mean P(correct type): **0.108**
- mean P(correct location): **0.134**

### performance_degradation

- mean P(correct type): **0.834**
- mean P(correct location): **0.086**

### device_failure

- mean P(correct type): **0.009**
- mean P(correct location): **0.000**

---

## Interpretation / decision tree

```
Evidence sufficient (9.10)
        ↓
Diagnosis = type × location joint discrete action (~72 on Star)
        ↓
Scan opens ~72 terminal diagnoses immediately
        ↓
Reward: exact match only full credit; partial credit is location-biased
        ↓
PPO learns type-only OR location-only shortcuts (9.9/9.11)
        ↓
Sparse joint classification problem — NOT fault-specific missing info
```

### Before intervention

- Do **not** apply fault-specific reward shaping (M type bonus, P interface bonus, etc.).
- Prefer **generic** fixes applied uniformly: partial type reward, factorized diagnosis actions,
  or curriculum — each modifies benchmark semantics; tag as protocol variant.
- Blind 100k alone is unlikely to fix joint type×location binding without structural change.

### Next decision (after confirming this audit)

1. Generic partial type+location reward (uniform all faults)
2. Factorized diagnosis action (type → location)
3. Restrict post-scan diagnosis until diagnostics used
4. Pivot research framing toward RL failure-mode analysis
