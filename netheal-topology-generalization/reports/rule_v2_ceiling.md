# Rule v2 ceiling (topology-agnostic)

> GT-free. No Star `center` / spoke expectations.
> Nominal edges from network graph skeleton vs post-scan discovery CONNECTED.
> Phase 1 of `GNN_expert_guided_spec.md`.

- Decision: **go**
- Star thresholds met: True; Mesh≥70%: True; policy blocked (Mesh&lt;50%): False

## Exact by setting

| Setting | Exact | Shortcut | Timeout | AvgSteps | Misses |
|---|---:|---:|---:|---:|---:|
| star_6 | 100% | 0% | 0% | 5.9 | 0 |
| star_8 | 99% | 0% | 1% | 6.4 | 1 |
| star_10 | 99% | 0% | 1% | 6.9 | 1 |
| linear_6 | 100% | 0% | 0% | 5.9 | 0 |
| mesh_6 | 100% | 0% | 0% | 5.8 | 0 |

## Per-fault (selected)

### star_6
- Device 100% (25/25; cons 25)
- Perf 100% (25/25; cons 21)
- Misconfig 100% (25/25; cons 25)
- Link 100% (25/25; cons 0)

### star_8
- Device 100% (25/25; cons 25)
- Perf 96% (24/25; cons 22)
- Misconfig 100% (25/25; cons 25)
- Link 100% (25/25; cons 0)
- Sample misses (1/1):
  - GT performance_degradation@edge_5->center → None@None (15 steps)

### star_10
- Device 100% (25/25; cons 25)
- Perf 96% (24/25; cons 23)
- Misconfig 100% (25/25; cons 25)
- Link 100% (25/25; cons 0)
- Sample misses (1/1):
  - GT performance_degradation@edge_8->center → None@None (15 steps)

### linear_6
- Device 100% (25/25; cons 25)
- Perf 100% (25/25; cons 20)
- Misconfig 100% (25/25; cons 25)
- Link 100% (25/25; cons 0)

### mesh_6
- Device 100% (25/25; cons 25)
- Perf 100% (25/25; cons 21)
- Misconfig 100% (25/25; cons 25)
- Link 100% (25/25; cons 0)

## Go / No-Go (spec)

- Star N=6 ≥95%, N=8/10 ≥90%: required for Graph BC Phase 3
- Mesh ≥70%: full go; 50–70%: go_with_mesh_caveat (document failure modes); &lt;50%: **no_go_fix_expert** (do not train Graph policy yet)

Run: `python -m src.run_rule_v2_ceiling`
