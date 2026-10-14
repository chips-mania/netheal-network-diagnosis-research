# Graph expert BC — B1 vs P1 (Phase 3)

> 역사적 PoC 기록. B1 Mesh는 실행하지 않았으므로 기존 0%/100% 표기는 실측이 아닌 placeholder였습니다. 아래를 N/A로 정정하며 원본 JSON은 보존합니다. 최신 공정 비교는 [재실험 보고서](../results/reliability_v2_20261008/report.md)를 참조하세요.

> Train: Rule **v2** demos Star N∈{4,5,6}.
> B1: pad12 MultiInputPolicy BC. P1: GNN+pointer BC.
> Eval: Seen Star6, Unseen scale Star8/10, Unseen family Mesh6.
> Decision: **go_scale_only**

- Flat transitions: 10926; Graph transitions: 10947
- B1 last loss: 0.0782; P1 last loss: 0.0462
- H1 scale (Star8 Δ): +14%; H2 mesh Δ: N/A (B1 미평가)
- Seen Star6 P1≥90%: True

## Exact

| Model | Setting | Exact | Shortcut | Timeout |
|---|---|---:|---:|---:|
| B1 MLP | star_6 | 89% | 0% | 5% |
| P1 GNN+ptr | star_6 | 100% | 0% | 0% |
| B1 MLP | star_8 | 76% | 0% | 7% |
| P1 GNN+ptr | star_8 | 90% | 0% | 10% |
| B1 MLP | star_10 | 57% | 0% | 10% |
| P1 GNN+ptr | star_10 | 42% | 0% | 58% |
| B1 MLP | mesh_6 | N/A | N/A | N/A |
| P1 GNN+ptr | mesh_6 | 9% | 0% | 70% |

## Per-fault P1

### star_6 (P1)
- Device 100% (25/25; cons 25)
- Perf 100% (25/25; cons 21)
- Misconfig 100% (25/25; cons 25)
- Link 100% (25/25; cons 0)

### star_8 (P1)
- Device 100% (25/25; cons 25)
- Perf 96% (24/25; cons 21)
- Misconfig 64% (16/25; cons 16)
- Link 100% (25/25; cons 0)

### star_10 (P1)
- Device 68% (17/25; cons 17)
- Perf 100% (25/25; cons 24)
- Misconfig 0% (0/25; cons 0)
- Link 0% (0/25; cons 0)

### mesh_6 (P1)
- Device 36% (9/25; cons 9)
- Perf 0% (0/25; cons 0)
- Misconfig 0% (0/25; cons 0)
- Link 0% (0/25; cons 0)

## Go / No-Go

- decision: **go_scale_only**
- h1_scale: True; h2_mesh: False

Run: `python -m src.run_graph_expert_bc`
