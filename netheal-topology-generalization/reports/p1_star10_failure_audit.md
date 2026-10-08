# P1 Star scale failure audit (N=6 / 8 / 10)

> Checkpoint: `models/checkpoints/graph_pointer_p1.pt`
> Rollout: P1 acting. Heads/ranks: Rule-v2 demo states (expert trajectory).
> Read: Many N=10 timeouts look like **commit/continue failures**: Rule-style conclude already matches GT from the same evidence mem, but the policy did not diagnose.

## 1. Online rollout (Exact / Timeout / Evidence→Commit)

| N | Exact | Timeout | Submit | Conclude=GT (any) | Ev→no-commit | Φ≥1 (PBRS) |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 100% | 0% | 100% | 97% | 0% | 72% |
| 8 | 87% | 13% | 87% | 100% | 13% | 64% |
| 10 | 46% | 54% | 46% | 81% | 35% | 49% |

### Per-fault @ N=10

| Fault | Exact | Timeout | Conclude=GT | Conclude|timeout | Missing∋GT | Ev→no-commit |
|---|---:|---:|---:|---:|---:|---:|
| device_failure | 84% | 16% | 84% | 0% | 0% | 0% |
| link_failure | 0% | 100% | 100% | 100% | 100% | 100% |
| misconfiguration | 0% | 100% | 40% | 40% | 0% | 40% |
| performance_degradation | 100% | 0% | 100% | 0% | 0% | 0% |

### Same @ N=8 (Misconfig early weakness check)

| Fault | Exact | Timeout | Conclude=GT | Ev→no-commit |
|---|---:|---:|---:|---:|
| device_failure | 100% | 0% | 100% | 0% |
| link_failure | 100% | 0% | 100% | 0% |
| misconfiguration | 48% | 52% | 100% | 52% |
| performance_degradation | 100% | 0% | 100% | 0% |

## 2. Expert-state head accuracy (Rule demos → P1 logits)

| N | Kind | Tool-kind | Diag-kind | Node | Edge | Fault |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 100% | 100% | 100% | 100% | 100% | 100% |
| 8 | 100% | 100% | 98% | 98% | 59% | 100% |
| 10 | 93% | 100% | 52% | 92% | 65% | 100% |

## 3. Correct-target rank (1=best) / Top-3

| N | Node mean-rank | Node top-3 | Edge mean-rank | Edge top-3 | Fault mean-rank |
|---:|---:|---:|---:|---:|---:|
| 6 | 1.00 | 100% | 1.00 | 100% | 1.00 |
| 8 | 1.04 | 100% | 3.35 | 72% | 1.00 |
| 10 | 1.22 | 95% | 2.68 | 67% | 1.00 |

## 4. Timeout trajectory samples (N=10)

- **link_failure** `edge_5->center` steps=15 submitted=False conclude_match=True
  - kinds: `scan → check_status → check_status → check_interfaces → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status`
  - end evidence: conclude_match=True missing_hit=True type=True loc=True
- **link_failure** `center->edge_7` steps=15 submitted=False conclude_match=True
  - kinds: `scan → check_status → check_status → check_interfaces → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status`
  - end evidence: conclude_match=True missing_hit=True type=True loc=True
- **device_failure** `edge_4` steps=15 submitted=False conclude_match=False
  - kinds: `scan → ping → ping → ping → ping → ping → ping → ping → ping → ping → ping → ping → ping → ping → ping`
  - end evidence: conclude_match=False missing_hit=False type=False loc=False
- **device_failure** `edge_4` steps=15 submitted=False conclude_match=False
  - kinds: `scan → ping → ping → ping → ping → ping → ping → ping → ping → ping → ping → ping → ping → ping → ping`
  - end evidence: conclude_match=False missing_hit=False type=False loc=False
- **misconfiguration** `edge_4` steps=15 submitted=False conclude_match=False
  - kinds: `scan → check_status → check_status → check_status → check_interfaces → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status`
  - end evidence: conclude_match=False missing_hit=False type=True loc=True
- **misconfiguration** `edge_4` steps=15 submitted=False conclude_match=False
  - kinds: `scan → check_status → check_status → check_status → check_interfaces → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status`
  - end evidence: conclude_match=False missing_hit=False type=True loc=True

## 5. Audit readout

- Edge pointer top-1 Δ N6→N10: **-35%** (top-3 @N10=67%, mean-rank=2.68)
- Diagnose-kind Δ N6→N10: **-48%**
- N10 overall evidence→no-commit: **35%**; conclude=GT on timeouts: **65%**

### Branch-level read

- **Link @N10**: Exact 0%, timeout 100%, conclude=GT on timeout **100%**, traj = status spam after iface — pure **commit failure**.
- **Misconfig @N8**: already Exact 48% / timeout 52% with conclude=GT **100%** on those eps — commit failure starts before N=10.
- **Misconfig @N10**: conclude=GT only ~40% — mix of incomplete asymmetry evidence + no commit.
- **Perf**: remains Exact 100% through N=10.
- **Expert-state diagnose-kind** collapses N6→N10 (kind still strong for tools) — when-to-diagnose is scale-fragile; edge top-1 falls by N=8 while top-3 stays ~2/3 (pointer calibration).

Many N=10 timeouts look like **commit/continue failures**: Rule-style conclude already matches GT from the same evidence mem, but the policy did not diagnose.

Run: `python -m src.run_p1_star10_failure_audit`
