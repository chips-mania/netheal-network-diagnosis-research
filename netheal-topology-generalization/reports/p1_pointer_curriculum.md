# Pointer curriculum (post-scan cue → target node)

> Train N∈{4,5,6} only. Oversample post-scan missing→endpoint node labels. No feature change. No N=8/10 in train.
> Decision: **curriculum_insufficient_try_features**

- cue demos/n≈400, copies×4, w_cue=3.0, epochs=50
- ckpt: `C:/Users/jaewon/Desktop/Project/NW_AI/NW_AI/netheal-topology-generalization/models/checkpoints/graph_pointer_cue_curriculum.pt` last_loss=2.2556

## First-cue spoke GT rank (forced scan)

| Model | N | top-1 | top-3 | mean-rank |
|---|---:|---:|---:|---:|
| baseline_joint | 6 | 100% | 100% | 1.00 |
| baseline_joint | 8 | 88% | 100% | 1.23 |
| baseline_joint | 10 | 13% | 57% | 3.07 |
| generic_joint | 6 | 100% | 100% | 1.00 |
| generic_joint | 8 | 37% | 100% | 1.63 |
| generic_joint | 10 | 0% | 22% | 5.87 |
| node_only | 6 | 100% | 100% | 1.00 |
| node_only | 8 | 88% | 100% | 1.12 |
| node_only | 10 | 20% | 87% | 2.37 |
| cue_curriculum | 6 | 100% | 100% | 1.00 |
| cue_curriculum | 8 | 12% | 100% | 2.51 |
| cue_curriculum | 10 | 26% | 33% | 3.74 |

## First-cue all GT

| Model | N | top-1 | top-3 | mean-rank |
|---|---:|---:|---:|---:|
| baseline_joint | 6 | 100% | 100% | 1.00 |
| baseline_joint | 8 | 90% | 100% | 1.20 |
| baseline_joint | 10 | 20% | 60% | 2.90 |
| node_only | 6 | 100% | 100% | 1.00 |
| node_only | 8 | 90% | 100% | 1.10 |
| node_only | 10 | 26% | 88% | 2.26 |
| cue_curriculum | 6 | 100% | 100% | 1.00 |
| cue_curriculum | 8 | 10% | 100% | 2.44 |
| cue_curriculum | 10 | 24% | 38% | 3.60 |

## Misconfig Exact (50 eps)

| Model | N | Exact | D incomplete | C commit | iface@GT |
|---|---:|---:|---:|---:|---:|
| baseline | 8 | 50% | 0% | 50% | 68% |
| baseline | 10 | 0% | 42% | 58% | 24% |
| curriculum | 8 | 24% | 20% | 0% | 0% |
| curriculum | 10 | 20% | 16% | 0% | 0% |

## Readout

- Spoke@N10 top-1: Generic 0% → node_only 20% → curriculum 26% (top-3 33%)
- Decision: **curriculum_insufficient_try_features**

If curriculum_insufficient → next: observation-derived missing-incident node features (still no GT leak).

Run: `python -m src.run_p1_pointer_curriculum`
