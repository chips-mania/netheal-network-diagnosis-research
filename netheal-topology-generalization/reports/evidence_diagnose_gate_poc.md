# Evidence diagnose-gate PoC

> Diagnosis locked until fault-like bits in obs (DOWN / latency≥15ms / discovery=2 / failed tool).
> Healthy dummy ping does not unlock. Reward after unlock is original. Named variant.
> Smoke: {'locked_after_scan': True, 'open_after_fail_ping': True, 'dummy_healthy_ping_locked': True}

| Metric | Control-Obs | Evidence-gate |
|---|---:|---:|
| Exact DSR | 11% | 0% |
| Shortcut | 100% | 0% |
| AvgSteps | 2.0 | 12.6 |
| Timeout | 0% | 76% |
| Submit | 100% | 24% |
| Device | 0% (0/25; cons 0) | 0% (0/25; cons 0) |
| Perf | 0% (0/25; cons 0) | 0% (0/25; cons 0) |
| Misconfig | 0% (0/25; cons 0) | 0% (0/25; cons 0) |
| Link | 44% (11/25; cons 0) | 0% (0/25; cons 0) |

- Train-time exact∧cons\|tool: 1.2%
- Decision: **no_go**
- Go if shortcut<10% and Device or Misconfig exact>=2 & cons>=2, not timeout/dummy
