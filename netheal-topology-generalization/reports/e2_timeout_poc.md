# E2 — Timeout-as-Incorrect-Diagnosis (seed 1)

> One change only: timeout without diagnosis gets the original incorrect-diagnosis penalty.
> Exact/wrong/partial on diagnose actions unchanged. Not mixed with E1/Gate/Factorized/Symmetric.
> Graded/symmetric partial was already tested in 9.13-A and is not repeated here.

Train: Star, 6 devices, 4 faults, MaskablePPO, 30000, seed 1
Eval: 100 ep, seed base 99000
Unit tests: {'diagnose_no_extra': 'ok', 'timeout_gets_incorrect_penalty': 'ok', 'expected': -16.0}

| Metric | Control | E2 |
|---|---:|---:|
| Exact DSR | 10% | 18% |
| **Non-link DSR** | 5% | 8% |
| Type Acc | 31% | 37% |
| Loc Acc | 55% | 61% |
| Shortcut Rate | 100% | 100% |
| AvgSteps | 2.0 | 2.0 |

| Fault | Control | E2 |
|---|---:|---:|
| link_failure | 24% | 48% |
| device_failure | 0% | 0% |
| performance_degradation | 16% | 24% |
| misconfiguration | 0% | 0% |

## Go/No-Go

- Non-link DSR Δ: **+2.7%**
- Shortcut Δ: **+0.0%**
- Decision: **no_go**
- No-Go if non-link DSR Δ < +3%p AND shortcut Δ > −10%p

## Traces

- `results/e2_timeout_poc_control_trace.csv`
- `results/e2_timeout_poc_variant_trace.csv`
