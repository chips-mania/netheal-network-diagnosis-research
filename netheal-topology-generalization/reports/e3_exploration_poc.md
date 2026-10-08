# E3 — Exploration Collapse + ent_coef=0.01 (seed 1)

> Reward/action unchanged. E1/E2 not mixed. Success = tool-use trajectories, not DSR.
> Go/No-Go uses **stochastic** eval. Deterministic DSR is reference only.

Train: Star, 6 devices, MaskablePPO, 30000, seed 1
Control `ent_coef=0.0` (SB3 default). E3 `ent_coef=0.01`.

## Stage A — Control action mix (stochastic probes)

| t | entropy_loss | sb3 ep_len | avg_steps | P(tool\|post-scan) | P(dx\|post-scan) | tool-use NL |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | None | None | 3.38 | 0.58 | 0.42 | 83% |
| 5120 | -3.398019030690193 | None | 2.88 | 0.37 | 0.53 | 17% |
| 10240 | -3.352712566405535 | None | 3.25 | 0.53 | 0.44 | 42% |
| 20480 | -3.124978866428137 | None | 3.12 | 0.50 | 0.47 | 58% |
| 30720 | -2.707691026851535 | None | 3.25 | 0.56 | 0.44 | 67% |

## Stage B — E3 probes

| t | entropy_loss | sb3 ep_len | avg_steps | P(tool\|post-scan) | P(dx\|post-scan) | tool-use NL |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | None | None | 3.38 | 0.58 | 0.42 | 83% |
| 5120 | -3.4214777782559396 | None | 3.12 | 0.44 | 0.47 | 50% |
| 10240 | -3.321948625892401 | None | 3.31 | 0.54 | 0.43 | 67% |
| 20480 | -3.1396287798881533 | None | 3.00 | 0.47 | 0.50 | 58% |
| 30720 | -2.849988343566656 | None | 3.25 | 0.53 | 0.44 | 50% |

## Final comparison

| Metric | Control sto | E3 sto | Control det | E3 det |
|---|---:|---:|---:|---:|
| AvgSteps | 2.58 | 2.74 | 2.00 | 2.00 |
| Tool-use before dx (non-link) | 35% | 43% | 0% | 0% |
| P(tool\|post-scan) | 0.35 | 0.42 | 0.00 | 0.00 |
| P(dx\|post-scan) | 0.64 | 0.58 | 1.00 | 1.00 |
| Exact DSR | 6% | 13% | 20% | 17% |
| Shortcut (det-style) | 66% | 54% | 100% | 100% |

## Go/No-Go (stochastic)

- E3 AvgSteps: **2.74** (need ≥ 3.0)
- E3 non-link tool-use: **43%** (need ≥ 10%)
- Control already: AvgSteps 2.58, non-link tool-use **35%**
- Precommitted rule: **go** (E3 tool-use ≥ 10%)
- No-Go if AvgSteps < 3.0 AND non-link tool-use-before-dx < 10% (stochastic)

## Stage A 해석

Entropy는 바닥이 아니다 (`entropy_loss` −3.40 → −2.71, H≈2.7–3.4 nats).  
**그리고** Control의 `P(tool|post-scan)`도 학습 내내 0.37–0.58이다. 스펙 표의 “entropy는 중간인데 tool 확률이 0”이 **아니다**.

학습/stochastic policy는 도구를 이미 샘플한다. 죽은 것은 **greedy(deterministic) mode**다.

| 모드 | AvgSteps | P(tool\|post-scan) | Shortcut |
|---|---:|---:|---:|
| Control train/stochastic | 2.6–3.4 | 0.35–0.56 | 66% (final sto) |
| Control deterministic | 2.0 | 0.00 | 100% |

이전 E1/E2의 “policy가 도구 state를 안 밟아서 shaping이 안 먹힌다”는 **deterministic eval 기준**이었다. PPO rollout은 stochastic이라, 학습 중에는 도구가 이미 나온다. 보상 개입이 실패한 이유는 “방문 없음”이 아니라 **argmax가 계속 `scan → diagnose`**인 것이다.

## E3가 한 일 / 안 한 일

`ent_coef=0.01`은 Control 대비 stochastic tool-use를 35%→43%, AvgSteps 2.58→2.74로만 움직였다.  
deterministic eval은 둘 다 AvgSteps 2.0, shortcut 100%, `P(tool|post-scan)=0`.

DSR은 성공 조건이 아니다 (참고: sto 6%→13%, det 20%→17%). E3 det Link DSR 68% / Device·Misconfig 0% — 숏컷 패턴 유지.

## Seed 2/3 · Stage C

사전 규칙은 Go라 seed 2/3를 **열 수는** 있다. 다만 Control이 이미 같은 바(tool-use ≥ 10%)를 넘겼으므로, 추가 seed는 “탐색이 열리는가”가 아니라 `ent_coef` Δ의 크기만 본다. 효과가 작다.

Stage C(PBRS on open exploration)는 **지금 올리지 않는다.** 학습 중 탐색은 Control에서도 이미 열려 있었고, 그 위에서 E1이 실패했다. 다음 레버는 ent_coef를 올리는 것이 아니라 **greedy가 도구를 고르게 만드는 것** (초반 diagnose-mask curriculum). `ent_coef=0.05`는 돌리지 않는다.
