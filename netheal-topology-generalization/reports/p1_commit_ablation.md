# Commit ablation: Baseline vs Generic vs Current

> Baseline: plain P1 BC.  
> **Generic**: warm-start + diagnose ×3 + w_diag=4, **w_hard=1**.  
> **Current**: same + **w_hard=2** on misconfig/link diagnose (prior run).
> Decision: **generic_sufficient_for_N10_link; hard_helps_N8_misconfig**

**N=10:** Generic Exact **66%** ≈ Current **67%** (Baseline 42%). Link **0%→92%** without fault-specific ×2 — the main +25%p story is a **generic commit objective**.  
**N=8:** Generic Exact **86%** < Current **96%** / Baseline **90%**, driven by Misconfig **48%** vs Current **88%** — hard×2 still helps Misconfig commit at near-scale; do not claim Generic fully replaces Current on every axis.

- Generic ckpt: `C:/Users/jaewon/Desktop/Project/NW_AI/NW_AI/netheal-topology-generalization/models/checkpoints/graph_pointer_p1_commit_generic.pt` last loss=0.0001

## Exact / Timeout

| N | Baseline Exact | Generic | Current | Generic TO | Current TO |
|---:|---:|---:|---:|---:|---:|
| 6 | 100% | 100% | 99% | 0% | 0% |
| 8 | 90% | 86% | 96% | 5% | 4% |
| 10 | 42% | 66% | 67% | 32% | 33% |

## Per-fault Exact @ N=8

| Fault | Baseline | Generic | Current |
|---|---:|---:|---:|
| device_failure | 100% | 100% | 100% |
| performance_degradation | 96% | 96% | 96% |
| misconfiguration | 64% | 48% | 88% |
| link_failure | 100% | 100% | 100% |

## Per-fault Exact @ N=10

| Fault | Baseline | Generic | Current |
|---|---:|---:|---:|
| device_failure | 68% | 72% | 68% |
| performance_degradation | 100% | 100% | 92% |
| misconfiguration | 0% | 0% | 8% |
| link_failure | 0% | 92% | 100% |

## Generic per-fault detail

### star_6
- Device 100% (25/25; cons 25)
- Perf 100% (25/25; cons 21)
- Misconfig 100% (25/25; cons 25)
- Link 100% (25/25; cons 0)

### star_8
- Device 100% (25/25; cons 25)
- Perf 96% (24/25; cons 21)
- Misconfig 48% (12/25; cons 12)
- Link 100% (25/25; cons 0)

### star_10
- Device 72% (18/25; cons 18)
- Perf 100% (25/25; cons 24)
- Misconfig 0% (0/25; cons 0)
- Link 92% (23/25; cons 0)

## Go / claim

- Auto-tag was `generic_commit_sufficient` (N10 Exact within 8%p of Current + Link≥80%).
- **Refined:** `generic_sufficient_for_N10_link; hard_helps_N8_misconfig`
- Link@N10 Generic=92% Current=100%
- N10 Exact Generic=66% Current=67% Baseline=42%
- N8 Misconfig Generic=48% Current=88% Baseline=64%

Claim for papers: *generic diagnose-commit supervision recovers the N=10 Link commit failure*; fault-specific diagnose upweight is **not** required for that gain, but still improves N=8 Misconfig.

Run: `python -m src.run_p1_commit_ablation`
