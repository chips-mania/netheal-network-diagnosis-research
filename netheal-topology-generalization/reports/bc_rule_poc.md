# BC on Rule trajectories

> Demos from GT-free RuleAgent (scan → disambiguate tools → diagnose).
> Not the GT 3-step expert. Policy never sees GT. Evidence obs ON, gate OFF.

- Demos kept: **2000** / tries 2000 (11675 transitions, avg len 5.8)
- Per-fault demos: {'misconfiguration': 491, 'link_failure': 496, 'performance_degradation': 497, 'device_failure': 516}
- Last BC loss: 0.0155

| Metric | BC-rule | (ref BC-informed) |
|---|---:|---:|
| Exact DSR | 100% | 58% |
| Shortcut | 0% | 0% |
| AvgSteps | 5.9 | 3.3 |
| Timeout | 0% | 0% |
| Device | 100% (25/25; cons 25) | 8% |
| Perf | 100% (25/25; cons 22) | 28% |
| Misconfig | 100% (25/25; cons 25) | 96% |
| Link | 100% (25/25; cons 0) | 100% |

- Decision: **go** (same hard-fault rule as gate)
