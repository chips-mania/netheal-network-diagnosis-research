# P1 train audit (tiny → heads → P0)

> Bottleneck read: **unclear_or_ok**
> Next: Increase P1 capacity/epochs with fixed recipe; target Seen≥90%

## Step 1 — Tiny overfit (200 transitions, 300 ep)

- Last loss: **0.0030**
- Joint required-head acc: **100.0%**
- Kind / Node / Edge / Fault: 100.0% / 1.0 / 1.0 / 1.0
- Strict pass (loss<0.05 & joint≥98%): **True**
- Soft pass: **True** → `architecture_can_overfit`

### By kind (tiny)

- check_interfaces: n=7, kind_acc=100.0%
- check_status: n=55, kind_acc=100.0%
- diagnose: n=35, kind_acc=100.0%
- ping: n=67, kind_acc=100.0%
- scan: n=36, kind_acc=100.0%

## Step 2 — Per-head on ~N=6 Rule-v2 set (80 ep)

- Transitions: 2301, last loss **0.0509**
- Joint: **99.4%**
- Kind: 100.0%
- Node pointer: 0.9858490566037735
- Edge pointer: 0.9981006647673314
- Fault type: 1.0

### By kind

- check_interfaces: n=104, kind_acc=100.0%
- check_status: n=530, kind_acc=100.0%
- diagnose: n=400, kind_acc=100.0%
- ping: n=867, kind_acc=100.0%
- scan: n=400, kind_acc=100.0%

## Step 3 — P0 GNN+flat

- Action space n=589
- Tiny(200): loss **0.3112**, acc **87.5%**, ok=False
- Full(~2211): loss **0.5347**, acc **79.9%**

## Readout

| Check | Result |
|---|---|
| P1 tiny overfit | PASS |
| P0 tiny overfit | FAIL (flat 589-way; not blocking P1) |
| Bottleneck | **cleared for P1** — proceed to Phase3 retrain |

### Fixes that unlocked tiny overfit

1. **Evidence mem** in `build_graph_obs(mem=…)` (ping/status/iface/missing/scanned) — graphs were nearly identical across steps without history.
2. **Collect graph before `choose`** so flags/queues are not mutated pre-snapshot.
3. **`id_rank` / edge src·dst rank** — breaks Star spoke isomorphism; Rule uses lex order that pure MP cannot see.

P0 tiny still fails because pad Discrete(~589) is a hard multi-class overfit; pointer factorization is the intended fix.

Next: `python -m src.run_graph_expert_bc` (Seen Star6 ≥90%, then scale/Mesh).

Run: `python -m src.run_p1_train_audit`
