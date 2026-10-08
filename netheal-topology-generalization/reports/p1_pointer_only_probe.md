# Pointer-only supervised probe

> Train: Rule node steps (status/iface/node-diagnose) on Star{4,5,6}; freeze kind/edge/fault heads.
> Eval: **forced-scan** first-cue GT node rank (kind-independent) + Rule-state ranks.
> Decision: **pointer_partial_scale_gap**

Node-only improves over joint but still incomplete at N=10 — capacity/curriculum still needed for full binding.

- ckpt: `C:/Users/jaewon/Desktop/Project/NW_AI/NW_AI/netheal-topology-generalization/models/checkpoints/graph_pointer_node_only.pt` last_loss=0.0000 n_train=4323

## Online first-cue — all GT

| Model | N | top-1 | top-3 | mean-rank |
|---|---:|---:|---:|---:|
| baseline_joint | 6 | 100% | 100% | 1.00 |
| baseline_joint | 8 | 82% | 100% | 1.36 |
| baseline_joint | 10 | 26% | 70% | 2.68 |
| generic_joint | 6 | 100% | 100% | 1.00 |
| generic_joint | 8 | 46% | 100% | 1.54 |
| generic_joint | 10 | 4% | 34% | 5.42 |
| node_only | 6 | 100% | 100% | 1.00 |
| node_only | 8 | 84% | 100% | 1.16 |
| node_only | 10 | 26% | 84% | 2.20 |

## Online first-cue — **spoke only** (critical)

| Model | N | top-1 | top-3 | mean-rank |
|---|---:|---:|---:|---:|
| baseline_joint | 6 | 100% | 100% | 1.00 |
| baseline_joint | 8 | 81% | 100% | 1.38 |
| baseline_joint | 10 | 23% | 69% | 2.75 |
| generic_joint | 6 | 100% | 100% | 1.00 |
| generic_joint | 8 | 44% | 100% | 1.56 |
| generic_joint | 10 | 0% | 31% | 5.60 |
| node_only | 6 | 100% | 100% | 1.00 |
| node_only | 8 | 83% | 100% | 1.17 |
| node_only | 10 | 23% | 83% | 2.25 |

## Offline Rule iface — pred=GT

| Model | N | pred=GT | GT top-1 |
|---|---:|---:|---:|
| baseline_joint | 6 | 100% | 100% |
| baseline_joint | 8 | 97% | 97% |
| baseline_joint | 10 | 83% | 83% |
| generic_joint | 6 | 100% | 100% |
| generic_joint | 8 | 100% | 100% |
| generic_joint | 10 | 100% | 100% |
| node_only | 6 | 100% | 100% |
| node_only | 8 | 100% | 100% |
| node_only | 10 | 100% | 100% |

## Readout

- Spoke@N10: Generic joint top-1=0% → node-only top-1=23% top-3=83%

Node-only improves over joint but still incomplete at N=10 — capacity/curriculum still needed for full binding.

If learnable: add node-pointer curriculum / loss weight on status·iface after missing cues.
If representation limit: enrich edge→endpoint features (e.g. missing-incident flags on nodes).

Run: `python -m src.run_p1_pointer_only_probe`
