# Star scale replay FT (+ N=4)

> Pad=12. Base Rule-BC on N=6. Fine-tune mixes **N=6 replay demos** with target-N demos.
> Contrast: previous few-shot FT without replay forgot N=6 and hurt Exact.

- Base demos: 2000, last loss 0.0123
- Replay N=6: 500 ep; target: 300 ep; FT epochs 15, lr=0.0001

## Summary (Exact DSR)

| N | Rule | Zero-shot (from N=6) | Replay-FT→8 | Replay-FT→10 | Replay-FT→4 |
|---:|---:|---:|---:|---:|---:|
| 4 | **100%** | ~0–22% | 7% | 4% | **77%** |
| 6 | **100%** | **99%** | **98%** | **98%** | **94%** |
| 8 | **99%** | **79%** | 72% | 70% | 62% |
| 10 | **98%** | **54%** | 64% | 53% | 41% |

## What changed vs naive FT

| | Naive FT (no N=6 demos) | Replay FT (500×N6 + 300×target) |
|---|---|---|
| N=6 after FT | 97% → 50% → 35% | **99% → 98% → 98%** |
| N=8 Exact | 66% (worse than zero 76%) | 72% (near zero 79%; retention fixed) |
| N=10 Exact | 45% | 53% |

**Replay does not magically beat zero-shot on larger N**, but it **stops catastrophic forgetting** of N=6.

## N=4

- Rule: **100%** — tree still closes.
- Zero-shot from N=6 policy: **fails** (~0–22%). Trace: loops `check_status(center)` / `ping(center→edge_2)` then wrong type — N=6 spoke habits.
- Replay-FT with **N=4 demos + N=6 replay**: Exact **77%**, N=6 retain **94%**.

Downward size transfer ≠ upward. Need target-size demos (with replay) for N=4.

## Per-fault highlights

### Zero-shot N=4 (main run)
- Exact 0%, shortcut 0% (all fault types 0/25)

### Replay-FT8 on N=8
- Exact 72%, Device 68%, Perf 72%, Misconfig 80%, Link 68%

### Replay-FT10 on N=10
- Exact 53%, Device 64%, Perf 40%, Misconfig 48%, Link 60%

### Retain N=6 after FT10
- Exact 98%, Device/Misconfig/Link 100%, Perf 92%

### Replay-FT4 → N=4
- Exact 77%, Device 64%, Perf 96%, Misconfig/Link in JSON `results/star_scale_replay_n4_extra.json`

## Runs

```text
python -m src.run_star_scale_replay
python -m src.run_star_scale_replay_n4
```

ckpts: `masked_ppo_bc_rule_pad12.zip`, `..._replay_ft8.zip`, `..._replay_ft10.zip`, `..._replay_ft4.zip`
