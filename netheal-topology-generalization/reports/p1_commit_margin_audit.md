# P1 diagnose commit margin audit

> Checkpoint: `C:/Users/jaewon/Desktop/Project/NW_AI/NW_AI/netheal-topology-generalization/models/checkpoints/graph_pointer_p1.pt`
> Question: on evidence-ready / timeout states, is diagnose 2nd–3rd (calibration) or ~0 (no commit)?
> Read: On conclude-ready states at N=10, diagnose often has **non-trivial mass / top-2 rank** → closer to **decision-boundary / calibration** than zero commit.

## Conclude-match steps (Rule `_try_conclude` == GT from same mem)

| N | n | P(diag) mean | median | margin mean | rank mean | diag top-1 | top-2 | top-3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 100 | 1.000 | 1.000 | 11.089 | 1.00 | 100% | 100% | 100% |
| 8 | 177 | 0.542 | 0.514 | 1.397 | 1.72 | 51% | 76% | 100% |
| 10 | 451 | 0.281 | 0.305 | -1.613 | 2.10 | 10% | 80% | 100% |

## Timeout final step

| N | n | P(diag) | margin | rank | diag top-1 | top-2 | dominant top kinds |
|---:|---:|---:|---:|---:|---:|---:|---|
| 6 | 0 | n/a | n/a | n/a | n/a | n/a | — |
| 8 | 9 | 0.160 | -4.397 | 2.44 | 0% | 56% | check_status:8, check_interfaces:1 |
| 10 | 54 | 0.149 | -7.258 | 2.44 | 0% | 56% | check_status:45, check_interfaces:5, ping:4 |

## Timeout ∩ conclude-match steps (smoking gun bucket)

| N | n | P(diag) | margin | rank | top-1 | top-2 |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0 | n/a | n/a | n/a | n/a | n/a |
| 8 | 86 | 0.184 | -3.167 | 2.49 | 0% | 51% |
| 10 | 405 | 0.200 | -2.567 | 2.22 | 0% | 78% |

## Success final step (sanity: should be diagnose)

| N | n | P(diag) | diag top-1 |
|---:|---:|---:|---:|
| 6 | 100 | 1.000 | 100% |
| 8 | 91 | 0.880 | 100% |
| 10 | 46 | 0.993 | 100% |

## Per-fault conclude-match @ N=10

| Fault | n | P(diag) | margin | top-1 | top-2 |
|---|---:|---:|---:|---:|---:|
| device_failure | 21 | 0.986 | 4.625 | 100% | 100% |
| link_failure | 300 | 0.265 | -1.449 | 0% | 87% |
| misconfiguration | 105 | 0.014 | -5.763 | 0% | 52% |
| performance_degradation | 25 | 1.000 | 8.603 | 100% | 100% |

## Readout

- Conclude-match P(diag) N6→N10: **1.000 → 0.281** (Δ -0.719)
- Conclude-match diagnose top-2 @N10: **80%**
- Timeout final P(diag) @N10: **0.149**; top-1 diagnose **0%**
- Timeout∩ready P(diag) @N10: **0.200**; top-2 **78%**

On conclude-ready states at N=10, diagnose often has **non-trivial mass / top-2 rank** → closer to **decision-boundary / calibration** than zero commit.

If near-zero on timeout-final → strengthen diagnose when evidence-ready (loss/curriculum), not just temperature.
If top-2 high but top-1 low → calibration / margin sharpening may suffice.

Run: `python -m src.run_p1_commit_margin_audit`
