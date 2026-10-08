# Repeat-action penalty PoC

> Train extra -2.0 when an action_id is reused in the episode.
> Stacked on evidence diagnose-gate + obs wrapper. Eval reward is original.
> Control = evidence-gate zip (no repeat penalty).

| Metric | Evidence-gate | +Repeat −2 |
|---|---:|---:|
| Exact DSR | 0% | 5% |
| Shortcut | 0% | 0% |
| AvgSteps | 12.7 | 6.0 |
| Timeout | 78% | 19% |
| Submit | 22% | 81% |
| Device | 0% (0/25; cons 0) | 0% (0/25; cons 0) |
| Perf | 0% (0/25; cons 0) | 20% (5/25; cons 5) |
| Misconfig | 0% (0/25; cons 0) | 0% (0/25; cons 0) |
| Link | 0% (0/25; cons 0) | 0% (0/25; cons 0) |

- Train-time exact∧cons\|tool: 1.6%
- Decision: **ambiguous**
- Go if shortcut<10% and Device or Misconfig exact>=2 & cons>=2, not timeout/dummy
