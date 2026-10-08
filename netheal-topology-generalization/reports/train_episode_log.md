# Train-time episode log — Control-Obs 30k

> Same protocol as Stage B. Reward/mask unchanged. Measurement only.
> JSONL: `results\train_episode_log\control_obs_star.jsonl`

| window | n | tool-use | exact | exact∧tool | exact\|tool | exact∧cons\|tool | shortcut |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 10499 | 43.5% | 3.9% | 1.5% | 3.4% | 0.4% | 53.4% |
| 0–10k | 3168 | 51.5% | 2.3% | 1.0% | 2.0% | 0.4% | 45.6% |
| 10–20k | 3332 | 45.4% | 3.9% | 1.6% | 3.6% | 0.4% | 51.1% |
| 20–30k | 3999 | 35.6% | 5.1% | 1.7% | 4.8% | 0.4% | 61.5% |

| fault | n | tool | exact | exact∧tool | exact∧cons |
|---|---:|---:|---:|---:|---:|
| device_failure | 2641 | 1153 | 11 | 5 | 1 |
| link_failure | 2700 | 1187 | 210 | 80 | 0 |
| misconfiguration | 2511 | 1073 | 15 | 5 | 0 |
| performance_degradation | 2647 | 1154 | 170 | 64 | 17 |
