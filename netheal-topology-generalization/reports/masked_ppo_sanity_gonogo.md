# Step 9.6-D — MaskablePPO Star Sanity

Eval episodes: 100
Training: Star only, 30000 steps, masked valid actions

## Results

Random-Valid                    DSR=0.040  TTD-success=15.00  AvgSteps=15.00  Cost=0.280  Cov=0.809  n=100
Masked PPO                      DSR=0.170  TTD-success=2.00  AvgSteps=2.00  Cost=0.300  Cov=0.000  n=100

## Behavior (Masked PPO)

- First-step diagnosis rate: 0.0%
- Invalid action rate: 0.0%
- scan usage: 100.0%
- ping usage: 0.0%
- Coverage: 0.000

### Fault-type DSR

| Fault | DSR |
|---|---:|
| link_failure | 68.0% |
| device_failure | 0.0% |
| performance_degradation | 0.0% |
| misconfiguration | 0.0% |

### Confusion matrix

| actual \ predicted | link_failure | performance_degradation |
| --- | --- | --- |
| device_failure | 25 | 0 |
| link_failure | 21 | 4 |
| misconfiguration | 25 | 0 |
| performance_degradation | 25 | 0 |

## GO Assessment (new criteria)

- ① DSR > Random-Valid (+5%p): **PASS** (margin=0.130)
- ② Troubleshooting behavior: **FAIL**
- ③ Action diversity (top pred <90%, invalid <5%): **FAIL**
- **Overall GO: NO**

## Next

- NO-GO: try 100k masked sanity, link-only curriculum, or observation/reward tuning — not Main Train.
- GO: checkpoint curve (10k/30k/50k/100k) then Step 10.
