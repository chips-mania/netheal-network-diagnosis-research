# Rule-BC seed + evidence-gate PPO

> 1) Short GT-free Rule BC (gate OFF while cloning).
> 2) MaskablePPO 30k with **evidence diagnose-gate**.
> Named variant. Original reward after unlock. Star N=6, pad=6.
> Smoke: locked_after_scan / open_after_fail_ping / dummy_healthy_ping_locked = all True.

- Seed: **400** demos, **15** BC epochs, last loss **2.10** (full BC-rule was ~0.015 @ 2000×40)
- PPO timesteps: 30000

## Exact / behavior

| Stage | Exact | Shortcut | Timeout | Submit | AvgSteps |
|---|---:|---:|---:|---:|---:|
| Seed BC (gate OFF) | **0%** | 0% | 46% | 54% | 11.8 |
| Seed+Gate PPO (gate ON) | **2%** | 0% | 50% | 50% | 10.1 |
| Gate-only PPO (prior) | ~0% | 0% | **76%** | ~22% | — |
| Full Rule-BC (prior) | 99% | 0% | 0% | 100% | 6.0 |

## Per-fault after Seed+Gate PPO

- Device 0% (0/25; cons 0)
- Perf 8% (2/25; cons 2)
- Misconfig 0% (0/25; cons 0)
- Link 0% (0/25; cons 0)

- Train exact∧cons|tool: ~3%
- Decision: **ambiguous** (shortcut 0%, but Exact≈0 and no Device/Misconfig hard wins; timeout 50% not full avoidance)

## Interpretation

1. **Short seed did not clone Rule.** Loss plateau ~2.1 and greedy Exact 0% → this PoC is *not* “good BC then gate PPO”; it is closer to weak init + gate.
2. vs **gate-only**: shortcut still 0%; timeout **50% vs 76%**, submit **50% vs 22%** → less pure avoidance, still no policy.
3. vs **full Rule-BC**: gap remains (~99% → 2%). Gate + light seed does not recover troubleshooting.

Run: `python -m src.run_bc_seed_gate_ppo`
