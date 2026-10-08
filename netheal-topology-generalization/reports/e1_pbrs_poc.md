# E1 — Pure Potential-Based Evidence Shaping (seed 1)

> Simulator ground truth was used only to compute the training-time shaping potential; it was not exposed to the policy observation.

Train: Star, 6 devices, 4 faults, MaskablePPO, 30000 steps, seed 1
Eval: 100 episodes, seed base 98000
α=1.0, γ=0.99. Eval shaping off. Evidence metrics = non-link only.

Unit tests: {'1': 'ok', '2': 'ok', '3': 'ok', '4': 'ok', '5': 'ok', '6': 'ok', '7': 'ok', '8': 'ok'}

## Primary (seed 1)

| Metric | Control (Masked) | E1 PBRS |
|---|---:|---:|
| Exact DSR | 7% | 7% |
| **Non-link DSR** | 5% | 5% |
| Type Acc | 28% | 28% |
| Loc Acc | 57% | 33% |
| Evidence Acquisition (non-link) | 0% | 0% |
| Evidence Consistency (non-link) | 0% | 0% |
| Shortcut Rate | 100% | 100% |
| AvgSteps | 2.0 | 2.0 |

| Variant | Link DSR | Link DSR Δ | Device | Perf | Misconfig |
|---|---:|---:|---:|---:|---:|
| Control | 12% | — | 0% | 16% | 0% |
| E1 | 12% | +0% | 0% | 16% | 0% |

## Go/No-Go (7.2)

- Non-link DSR Δ: **+0.0%** (No-Go if < +3%p)
- Evidence Acquisition Δ: **+0.0%** (No-Go if < +5%p)
- Decision: **no_go**
- No-Go if non-link DSR Δ < +3%p AND Evidence Acquisition Δ < +5%p

## Assessment

- Verdict: **no_go_seed1**
- Next: Dense credit alone did not move non-link / acquisition. Hold seed 2/3. Consider E2.

## Traces

- `results/e1_pbrs_poc_control_trace.csv`
- `results/e1_pbrs_poc_variant_trace.csv`
