# Full Rule-BC + evidence-gate PPO

> In-process: Rule-BC (pad=6) → MaskablePPO 30k with **evidence diagnose-gate**.
> Named variant. Original reward after unlock. Avoids cross-process action-id drift.
> Smoke: {'locked_after_scan': True, 'open_after_fail_ping': True, 'dummy_healthy_ping_locked': True}

- Demos: **2000**, BC epochs 40, last loss 0.0149
- PPO timesteps: 30000, pad=6

## Exact / behavior

| Stage | Exact | Shortcut | Timeout | Submit | AvgSteps |
|---|---:|---:|---:|---:|---:|
| Full Rule-BC (gate OFF) | 98% | 0% | 0% | 100% | 6.0 |
| Full Rule-BC zero-shot (gate ON) | 48% | 0% | 50% | 50% | 11.0 |
| Full Rule-BC + Gate PPO | 77% | 0% | 22% | 78% | 8.3 |
| Short-seed + Gate PPO (prior) | ~2% | 0% | 50% | 50% | 10.1 |
| Gate-only PPO (prior) | ~0% | 0% | **76%** | ~22% | — |

## Per-fault after Gate PPO

- Device 100% (25/25; cons 25)
- Perf 100% (25/25; cons 25)
- Misconfig 8% (2/25; cons 2)
- Link 100% (25/25; cons 0)

## Zero-shot under gate (before PPO)

- Device 100% (25/25; cons 25)
- Perf 92% (23/25; cons 23)
- Misconfig 0% (0/25; cons 0)
- Link 0% (0/25; cons 0)

- Train exact∧cons\|tool: 47.2%
- Decision: **go**
- Go if shortcut<10% and Device or Misconfig exact>=2 & cons>=2, not timeout/dummy

Run: `python -m src.run_bc_rule_gate_ppo`
