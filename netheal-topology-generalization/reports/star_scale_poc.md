# Star scale + pad12 fine-tune

> Obs/action padded to **max_devices=12** (canonical `center`/`edge_*`).
> Base: Rule-BC on Star N=6. FT: few-shot Rule-BC on N=8 then N=10.

- Base demos: **2000** (avg len 5.8), last BC loss 0.0148
- FT demos/epochs: 300 / 10 (lr=0.0001)
- ckpt: `models/checkpoints/masked_ppo_bc_rule_pad12.zip` (+ `..._ft8.zip`, `..._ft10.zip`)

## Summary

| N | Rule Exact | Zero-shot Exact | FT Exact | Shortcut (FT or zero) | AvgSteps |
|---:|---:|---:|---:|---:|---:|
| 6 | 100% | 97% | — | 0% | 5.9 |
| 8 | 97% | 76% | 66% | 0% | 6.3 |
| 10 | 99% | 64% | 45% | 0% | 6.1 |

## Interpretation

1. **Pad works.** Same Discrete(589) + obs `(12,·)` across N=6/8/10; N=6 pad12 BC reaches **97%** (vs prior unpadded BC-rule 99%).
2. **Rule ceiling holds** at larger Star (97–100%). Remaining misses are mostly step-budget timeouts, not logic failure.
3. **Zero-shot transfers partially** (76% @8, 64% @10), shortcut still 0%. New edges are partly handled via shared `center`/low-index actions + mask, but not full Rule performance.
4. **Naive few-shot FT hurt.** 300 demos × 10 epoch CE **without N=6 replay** dropped Exact below zero-shot and erased retention (97%→50%→35%). Catastrophic forgetting, not “FT is impossible.”
5. Next FT recipe (not run here): mix `base demos ∪ new-N demos` (replay), or longer FT with lower lr, or freeze early layers.

## Per-fault (selected)

### Base N=6
- Exact 97%, shortcut 0%, timeout 0%
- Device 100% (25/25; cons 25)
- Perf 88% (22/25; cons 20)
- Misconfig 100% (25/25; cons 25)
- Link 100% (25/25; cons 0)

### Zero-shot N=8
- Exact 76%, shortcut 0%, timeout 0%
- Device 76% (19/25; cons 19)
- Perf 68% (17/25; cons 16)
- Misconfig 80% (20/25; cons 20)
- Link 80% (20/25; cons 0)

### FT N=8
- Exact 66%, shortcut 0%, timeout 0%
- Device 44% (11/25; cons 11)
- Perf 68% (17/25; cons 17)
- Misconfig 68% (17/25; cons 17)
- Link 84% (21/25; cons 0)

### Zero-shot N=10
- Exact 64%, shortcut 0%, timeout 1%
- Device 72% (18/25; cons 18)
- Perf 76% (19/25; cons 16)
- Misconfig 40% (10/25; cons 10)
- Link 68% (17/25; cons 0)

### FT N=10
- Exact 45%, shortcut 0%, timeout 1%
- Device 24% (6/25; cons 6)
- Perf 32% (8/25; cons 8)
- Misconfig 64% (16/25; cons 16)
- Link 60% (15/25; cons 0)

## N=6 retention after FT
- retain6_after_8: Exact 50%
- retain6_after_10: Exact 35%

- FT N=8 hard-fault gate decision: **go** (Device cons≥2; overall Exact still worse than zero-shot)

Run: `python -m src.run_star_scale_poc`
