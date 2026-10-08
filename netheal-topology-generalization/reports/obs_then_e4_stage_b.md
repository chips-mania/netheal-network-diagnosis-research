# Stage A/B — Evidence observation wrapper + Control-Obs

> NetHeal package unchanged. Reward / mask curriculum unchanged. E4 not trained.
> Train: Star 4-fault MaskablePPO 30000 seed 1, wrapper ON.

## Stage A unit tests

| test | pass |
|---|:---:|
| 1_status_down | True |
| 2_ping_fail | True |
| 3_perf_latency | True |
| 4_healthy_ping | True |
| 5_raw_feat0_zero | True |

## Stage B probes (Control-Obs, retrained)

| probe | Φ | feat0 DOWN | max latency slot | ΔP(correct) | TV |
|---|---:|:---:|---:|---:|---:|
| device_edge0_ping_to_gt | 2 | True | 0.000 | +0.0000 | 0.034 |
| device_edge0_ping_healthy_pair | 0 | False | 0.062 | +0.0001 | 0.091 |
| device_edge0_check_status | 2 | True | 0.000 | +0.0000 | 0.044 |
| perf_center_edge0_ping | 2 | False | 0.206 | -0.0019 | 0.088 |
| misconfig_edge0_ping_to_gt | 0 | False | 0.054 | +0.0001 | 0.080 |

## Negative control (E3 zip + wrapper, no retrain)

| probe | Φ | ΔP(correct) | TV |
|---|---:|---:|---:|
| device_edge0_ping_to_gt | 2 | +0.0000 | 0.049 |
| device_edge0_ping_healthy_pair | 0 | +0.0002 | 0.086 |
| device_edge0_check_status | 2 | +0.0000 | 0.087 |
| perf_center_edge0_ping | 2 | -0.0110 | 0.093 |
| misconfig_edge0_ping_to_gt | 0 | +0.0002 | 0.076 |

## Stochastic NL (Control-Obs, n=200)

| Metric | E3 zip audit | Control-Obs |
|---|---:|---:|
| P(Φ≥1 \| tool-use) | 28.5% | 22.6% |
| P(exact ∧ cons \| tool-use) | 0.7% | 0.0% |
| P(exact ∧ cons \| Φ≥1) | 2.4% | 0.0% |

## Decision

- Device ΔP(correct): **+0.0000**
- Perf ΔP(correct): **-0.0019**
- Dummy ΔP(correct): **+0.0001**
- Hypothesis support: **False**
- Decision: **no_go** (open E4: False)
- Go if (Device or Perf ΔP>=+0.05 and > dummy) AND (P(ec|Φ)>=10% or ΔP>=0.05 & P(ec|tool)>=5%)

Stage C E4 was not implemented or trained.
