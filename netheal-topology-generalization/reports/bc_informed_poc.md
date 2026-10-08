# BC on informed trajectories

> Expert (GT only for demo gen): scan → evidence tool → exact diagnosis.
> Policy never sees GT. Evidence obs ON, diagnose-gate OFF (clone the 3-step path itself).

- Demos kept: **2000** / tries 2000 (6000 transitions)
- Per-fault demos: {'performance_degradation': 496, 'misconfiguration': 505, 'device_failure': 527, 'link_failure': 472}
- Last BC loss: 0.6902

| Metric | BC-informed |
|---|---:|
| Exact DSR | 58% |
| Shortcut | 0% |
| AvgSteps | 3.3 |
| Timeout | 0% |
| Device | 8% (2/25; cons 1) |
| Perf | 28% (7/25; cons 2) |
| Misconfig | 96% (24/25; cons 0) |
| Link | 100% (25/25; cons 0) |

- Decision: **ambiguous** (same hard-fault rule as gate)
