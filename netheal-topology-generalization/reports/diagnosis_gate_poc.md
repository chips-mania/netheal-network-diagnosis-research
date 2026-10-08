# Step 9.13-B — Diagnosis Gate Ablation

> **Protocol variant G** — diagnosis blocked until ≥1 diagnostic action.
> Ablation only; not proposed as the final topology-generalization protocol.

Train: Star, 6 devices, 4 faults, MaskablePPO, 30000 steps, seed 1
Eval: 100 episodes, seed base 96000

## Comparison

| Metric | Control (Masked) | Variant G (Gate) |
|---|---:|---:|
| Exact DSR | 19% | 10% |
| Type Accuracy | 23% | 32% |
| Location Accuracy | 48% | 28% |
| AvgSteps | 2.0 | 3.0 |
| Tools before dx | 1.0 | 2.0 |
| **Shortcut Rate** | 100% | 0% |
| Coverage | 0.73 | 0.73 |

### Per-fault DSR

| Fault | Control | Gate |
|---|---:|---:|
| link_failure | 76% | 28% |
| device_failure | 0% | 0% |
| performance_degradation | 0% | 12% |
| misconfiguration | 0% | 0% |

## Assessment

- DSR delta: **-9%**
- Shortcut: 100% → 0%
- Non-link faults improved (≥5%p): **1/3**
- Dummy-tool collapse: **True**
- Verdict: **dummy_tool_collapse**
- Next: Gate only forced a dummy diagnostic; proceed to 9.13-C Factorized Diagnosis.

## Traces

- `results/diagnosis_gate_poc_control_trace.csv`
- `results/diagnosis_gate_poc_variant_trace.csv`
