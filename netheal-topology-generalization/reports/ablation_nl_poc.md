# Ablation-NL — Non-link-only train distribution (seed 1)

> Reward / 72-way action space unchanged. GT sampler excludes link. E1/E2/E3 not mixed.
> Primary eval = non-link-only, same n/seeds for Control and Ablation-NL.
> Go/No-Go uses **deterministic** primary. Stochastic is recorded.

Train: Star, 6 devices, MaskablePPO, 30000 timesteps, seed 1
Control weights: masked_ppo_e3_control_star.zip
Confound: ~1.33× non-link exposure (timesteps, not episodes). Not separated from competition removal.

## Primary — non-link-only eval

| Metric | Control sto | Ablation-NL sto | Control det | Ablation-NL det |
|---|---:|---:|---:|---:|
| Non-link DSR | 4% | 4% | 9% | 9% |
| Device | 0% (0/34; cons 0) | 0% (0/34; cons 0) | 0% (0/34; cons 0) | 0% (0/34; cons 0) |
| Perf | 12% (4/33; cons 1) | 12% (4/33; cons 0) | 27% (9/33; cons 0) | 27% (9/33; cons 0) |
| Misconfig | 0% (0/33; cons 0) | 0% (0/33; cons 0) | 0% (0/33; cons 0) | 0% (0/33; cons 0) |
| Tool-use (pre-dx) | 43% | 34% | 0% | 0% |
| Evidence Acquisition | 9% | 9% | 0% | 0% |
| Evidence-Consistent | 11% | 0% | 0% | 0% |
| Shortcut | 55% | 63% | 100% | 100% |
| AvgSteps | 2.83 | 2.65 | 2.00 | 2.00 |

## Secondary — Ablation-NL on 4-fault mix (reference)

- Link DSR sto/det: 8% / 20%
- 4-fault Exact DSR sto/det: 3% / 11%
- Control 4-fault det Exact (zip check): 20% (E3 published 20%)

## Decision (deterministic NL eval)

- Hypothesis support: **False** faults=[]
- Non-link DSR Δ: **+0.0%** (need ≥ +5%p for Go)
- Shortcut/AvgSteps similar: True
- Perf-only up: False
- Decision: **no_go**
- Go if Device/Misconfig det exact>=2 and exact-consistent>=2 AND NL DSR +5%p (det)

## Interpretation

Control 4-fault det Exact **20%** — E3 zip identity OK.

NL-only det에서 Control과 Ablation-NL이 **숫자까지 같다**: Device 0/34, Misconfig 0/33, Perf 9/33, shortcut 100%, AvgSteps 2.0, evidence 0%. Link를 빼도 greedy는 `scan → diagnose`다.

9.9 prior와 맞다. mixture에서 Link 경쟁을 제거하는 것은 Device/Misconfig 0% 벽을 열지 못했다. Perf 27%는 Control에도 이미 있던 scan-only 복권이며, evidence-consistent 0%라 troubleshooting이 아니다.

Secondary: Ablation-NL의 4-fault Link DSR det 20% (Control 참고 56%). Link GT를 안 보고도 가끔 link를 맞춰지만, non-link는 안 올랐다.

다음: E4 diagnose-mask curriculum. seed 2/3 · Link-빈도 ablation · Stage C PBRS는 열지 않는다.
