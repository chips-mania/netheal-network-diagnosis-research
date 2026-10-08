# Misconfig cue→node pointer rank audit

> After `asym_from_GT` first appears: node-head rank of GT. Online = policy rollout; Offline = Rule trajectory states.
> After asym-missing cue @N10, GT spoke rank collapses (top-1 very low) → **cue→node binding** fails to generalize under scale.

## Online: first asym-missing cue — GT node rank

| Model | N | n | top-1 | top-3 | mean-rank | P(GT) | margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 6 | 50 | 100% | 100% | 1.00 | 0.976 | 6.665 |
| baseline | 8 | 50 | 76% | 100% | 1.48 | 0.603 | 1.664 |
| baseline | 10 | 50 | 18% | 60% | 2.92 | 0.204 | -1.831 |
| generic | 6 | 50 | 100% | 100% | 1.00 | 1.000 | 9.919 |
| generic | 8 | 50 | 48% | 100% | 1.52 | 0.497 | 1.468 |
| generic | 10 | 50 | 2% | 36% | 5.48 | 0.027 | -6.818 |

## Online: first cue — center vs spoke (Generic)

| N | Center top-1 | Spoke top-1 | Spoke top-3 | Spoke mean-rank |
|---:|---:|---:|---:|---:|
| 6 | 100% | 100% | 100% | 1.00 |
| 8 | 100% | 42% | 100% | 1.58 |
| 10 | 100% | 0% | 35% | 5.57 |

## Online: when policy chooses iface — picks GT?

| Model | N | n_iface | chose GT | GT top-1 in scores |
|---|---:|---:|---:|---:|
| baseline | 6 | 50 | 100% | 100% |
| baseline | 8 | 80 | 42% | 42% |
| baseline | 10 | 91 | 9% | 9% |
| generic | 6 | 50 | 100% | 100% |
| generic | 8 | 50 | 48% | 48% |
| generic | 10 | 46 | 2% | 2% |

## Offline Rule demos: first cue + expert iface steps (GT rank)

| Model | N | cue top-1 | cue rank | iface steps pred=GT | iface GT top-1 |
|---|---:|---:|---:|---:|---:|
| baseline | 6 | 0% | 3.12 | 100% | 100% |
| baseline | 8 | 0% | 4.47 | 85% | 85% |
| baseline | 10 | 0% | 4.78 | 82% | 82% |
| generic | 6 | 0% | 3.58 | 100% | 100% |
| generic | 8 | 0% | 4.03 | 100% | 100% |
| generic | 10 | 0% | 4.62 | 100% | 100% |

## Readout

After asym-missing cue @N10, GT spoke rank collapses (top-1 very low) → **cue→node binding** fails to generalize under scale.

### Sharp splits (Generic online)

| N | All GT top-1 | Spoke top-1 | Spoke top-3 | Center top-1 |
|---:|---:|---:|---:|---:|
| 6 | 100% | 100% | 100% | 100% |
| 8 | 48% | 42% | 100% | 100% |
| 10 | **2%** | **0%** | **35%** | 100% |

- N=8: still top-3=100% → ranking/calibration.
- N=10 spoke: top-3 only 35%, mean-rank ~5.6 → **binding collapse**, not thin margin.
- Center stays solvable → matches path-audit success bias.
- When policy issues iface @N10, chose GT **2%** — same failure.

Offline Rule-state iface pred=GT remains high (expert trajectory has richer mem); online first-cue is the scale-generalization stress test.

Next: pointer-only supervised probe on first-cue / Rule status-iface states.

Run: `python -m src.run_p1_misconfig_pointer_rank`
