# Step 9.13-A — Symmetric Partial Reward PoC

> **Protocol variant R** — symmetric type/location partial reward applied uniformly to all faults.
> Control uses original NetHeal `SparseRewardCalculator`.

Train: Star, 6 devices, 4 faults, MaskablePPO, 30000 steps, seed 1
Eval: 100 episodes, seed base 95000

## Comparison

| Metric | Control (Original) | Variant R (Symmetric) |
|---|---:|---:|
| Exact DSR | 19% | 4% |
| Type Accuracy | 42% | 25% |
| Location Accuracy | 49% | 9% |
| AvgSteps | 2.0 | 2.0 |
| Tools before dx | 1.0 | 1.0 |
| Coverage | 0.73 | 0.73 |

### Per-fault DSR

| Fault | Control | Variant R |
|---|---:|---:|
| link_failure | 68% | 0% |
| device_failure | 0% | 16% |
| performance_degradation | 8% | 0% |
| misconfiguration | 0% | 0% |

## Complementary failure check

- Control: Type 42% / Loc 49%
- Variant: Type 25% / Loc 9%

## Assessment

- DSR delta: **-15%**
- Complementary gap reduced: **False**
- Non-link faults improved (≥5%p): **1/3**
- Top single prediction share: **100%**
- Verdict: **inconclusive_or_fail**
- Next: Proceed to 9.13-B Diagnosis Gate ablation; do not scale to 100k yet.

## Traces

- `results/symmetric_reward_poc_control_trace.csv`
- `results/symmetric_reward_poc_variant_trace.csv`
