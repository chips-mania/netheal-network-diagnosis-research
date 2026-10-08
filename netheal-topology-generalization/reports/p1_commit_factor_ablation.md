# Commit factor ablation: weight-only (B) vs Generic

> **B**: warm-start + w_diag=4, oversample×1, w_hard=1  
> **Generic**: w_diag=4 + oversample×3 (prior)  
> Also reports ready-commit vs premature-diagnose rates.
> Decision: **weight_helps_oversample_adds**

Weight-only helps but undershoots Generic; oversampling still contributes to the full gain.

N10 ready_commit Baseline 9%→B 53%; premature|diag 0%→0%.

- B ckpt: `C:/Users/jaewon/Desktop/Project/NW_AI/NW_AI/netheal-topology-generalization/models/checkpoints/graph_pointer_p1_commit_wonly.pt` last_loss=0.0003

## Exact

| N | Baseline | B (weight-only) | Generic |
|---:|---:|---:|---:|
| 6 | 100% | 100% | 100% |
| 8 | 90% | 86% | 86% |
| 10 | 42% | 61% | 66% |

## Per-fault @ N=10 (B)

- Device 68% (17/25; cons 17)
- Perf 100% (25/25; cons 24)
- Misconfig 8% (2/25; cons 2)
- Link 68% (17/25; cons 0)

## Ready vs premature (same eval seeds)

| Model | N | Exact | Ready-commit | Premature\|diag | Premature ep |
|---|---:|---:|---:|---:|---:|
| baseline | 8 | 91% | 49% | 0% | 0% |
| baseline | 10 | 45% | 9% | 0% | 0% |
| B_wonly | 8 | 87% | 59% | 1% | 1% |
| B_wonly | 10 | 76% | 53% | 0% | 0% |
| generic | 8 | 87% | 59% | 4% | 4% |
| generic | 10 | 70% | 96% | 0% | 0% |

## Readout

- Link@N10 B=68% (Baseline 0%, Generic ~88%)
- N10 Exact B=61% Generic=66% Baseline=42%

Good intervention signature: ready-commit ↑, premature ≈ flat.

Run: `python -m src.run_p1_commit_factor_ablation`
