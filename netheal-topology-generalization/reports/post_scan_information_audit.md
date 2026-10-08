# Step 9.8 — Post-Scan Information Audit

## Setup

- Topology: Star, 6 devices
- Network seed (controlled comparison): 42
- Anchor locations: {"link_failure": "center->edge_0", "performance_degradation": "center->edge_0", "device_failure": "edge_0", "misconfiguration": "edge_0"}
- Probe dataset: 200 episodes (rotating 4 fault types, controlled anchor re-inject)

## 1. Controlled fault comparison (same seed, different fault type)

| Fault | Location | Discovery Δ cells | Post-scan disc nonzero |
|---|---|---:|---:|
| link_failure | center->edge_0 | 8 | 8 |
| device_failure | edge_0 | 10 | 10 |
| performance_degradation | center->edge_0 | 10 | 10 |
| misconfiguration | edge_0 | 10 | 10 |

### Post-scan L2 distance between fault types (same seed=42)

- link_failure__vs__device_failure: 1.4142
- link_failure__vs__performance_degradation: 1.4142
- link_failure__vs__misconfiguration: 1.4142
- device_failure__vs__performance_degradation: 0.0000
- device_failure__vs__misconfiguration: 0.0000
- performance_degradation__vs__misconfiguration: 0.0000

## 2. Scan reveals fault location?

After `scan_network`, discovery matrix encodes **UP connections only** (see `env._execute_topology_action` — failed links are omitted, not marked DISCONNECTED).

Implication: **link failure location** may appear as a **missing edge** among otherwise connected star edges. Device/performance/misconfig signals in scan-only obs are weaker or absent.

## 3. Scan-only supervised probe (fault type classification)

- Random chance (4-class): 25.00%

| Model | CV accuracy |
|---|---:|
| logistic_regression | 50.0% ± 0.0% |
| decision_tree | 50.0% ± 0.0% |
| random_forest | 50.0% ± 0.0% |

### Per-field logistic regression (post-scan)

| Field | CV accuracy |
|---|---:|
| discovery_matrix | 50.0% ± 0.0% |
| device_status | zero variance |
| recent_diagnostics | 25.0% ± 0.0% |
| episode_metadata | 25.0% ± 0.0% |

## 4. Interpretation

- Best probe: **logistic_regression** @ 50.0% (random chance 25%)
- Branch: **B_partial**

### Key finding (controlled seed=42)

| Post-scan observation | link_failure | device / perf / misconfig |
|---|---|---|
| Discovery matrix | **8 nonzero** (one edge missing) | **10 nonzero** (full star) |
| device_status | all zeros | all zeros |
| Pairwise L2 vs others | 1.41 | **0.00** (three faults identical) |

**Conclusion:**

1. **Scan-only reveals link failure location** via a missing edge in the discovery matrix (failed links are not marked CONNECTED during scan).
2. **Scan-only does NOT separate device / performance / misconfiguration** at the same anchor — post-scan observation vectors are identical.
3. Probe **50% ≈ binary(link vs non-link)** — not true 4-class separability.
4. This explains link-only PPO **100%** and multi-fault **link_failure collapse** without ping.

Some scan-only separability but not decisive for full 4-fault diagnosis. Multi-fault masked training may help for non-link faults; ping/status likely needed for device/perf/misconfig.

## 5. Next step

- Branch **A**: pivot metrics/questions toward diagnosis-mapping generalization (DSR, Macro F1, location accuracy).
- Branch **B**: masked multi-fault 100k sanity; ping/status may become necessary.
- Do **not** run link-only 50k/100k expecting ping to emerge.
