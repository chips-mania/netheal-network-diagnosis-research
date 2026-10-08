# Step 9.13-C — Factorized Diagnosis PoC

> **Protocol variant F** — diagnosis is type (4) then location (6 or 30).
> Structural protocol variant; does not modify the NetHeal package.

Train: Star, 6 devices, 4 faults, MaskablePPO, 30000 steps, seed 1
Eval: 100 episodes, seed base 97000

## Smoke (structure)

- Post-scan type actions: **4** (expect 4)
- After type-commit locations: **6** (expect 6 for `device_failure`)
- Type-commit terminates: **False**

## Comparison

| Metric | Control (Masked) | Variant F (Factorized) |
|---|---:|---:|
| Exact DSR | 12% | 0% |
| Type Accuracy | 12% | 0% |
| Location Accuracy | 55% | 0% |
| AvgSteps | 2.0 | 15.0 |
| Tools before dx | 1.0 | 15.0 |
| Shortcut Rate (scan→dx ≤2) | 100% | 0% |
| Factorized Shortcut (scan→type→loc ≤3) | 0% | 0% |
| Coverage | 0.74 | 0.00 |

### Per-fault DSR

| Fault | Control | Factorized |
|---|---:|---:|
| link_failure | 48% | 0% |
| device_failure | 0% | 0% |
| performance_degradation | 0% | 0% |
| misconfiguration | 0% | 0% |

## Assessment

- DSR delta: **-12%**
- Joint shortcut: 100% → 0%
- Factorized shortcut: **0%**
- Non-link faults improved (≥5%p): **0/3**
- Uses diagnostic tools (≥2 before dx): **False** (never diagnosed; 15.0 is timeout padding)
- Verdict: **diagnosis_avoidance_collapse**
- Next: Policy never submits type or location; timeout (-1.5) beats a wrong diagnosis (~-16). Do not scale 100k or Mesh on factorization alone.

## Policy behavior

Control remains **scan → diagnose** (joint 72-way lottery). Link DSR is the only non-zero cell.

Variant F **never commits a type and never submits a location**. Typical episode:

```
scan → ping(edge_1→edge_3) × N → timeout at 15 steps
```

This is locally rational under the original sparse reward:

| Outcome | Approx. return |
|---|---:|
| Exact diagnosis | +15.9 |
| Wrong diagnosis | −16.1 |
| Timeout (15 × −0.1) | **−1.5** |

Type-commit is unrewarded. The second action still faces a 6- or 30-way location lottery. Avoiding diagnosis entirely dominates guessing. Coverage 0.00 is a metrics-wrapper artifact; traces show `scan_network` on step 1 in every episode.

## Traces

- `results/factorized_diagnosis_poc_control_trace.csv`
- `results/factorized_diagnosis_poc_variant_trace.csv`
