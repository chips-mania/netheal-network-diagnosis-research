# GPT-6.1 Sol vs Jev API pilot

This is a development pilot, not the final confirmatory evaluation.
Costs are usage-based USD estimates; invoices have not been reconciled.
Run status: completed; verification: passed.

| Provider | Episodes | Exact | Wrong diagnosis | Mean seconds | p95 seconds | Mean USD | Total USD | Tools |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt | 64 | 87.5% | 12.5% | 21.07 | 43.07 | 0.048026 | 3.073663 | 5.45 |
| jev | 64 | 26.6% | 67.2% | 1.36 | 8.82 | 0.001562 | 0.099938 | 4.84 |

## Per condition

| Condition | GPT Exact | Jev Exact | GPT seconds | Jev seconds |
|---|---:|---:|---:|---:|
| linear_6 | 87.5% | 25.0% | 17.01 | 0.54 |
| mesh_10 | 100.0% | 25.0% | 22.97 | 3.64 |
| mesh_6 | 87.5% | 25.0% | 17.37 | 0.75 |
| random_10 | 100.0% | 25.0% | 32.27 | 3.15 |
| random_6 | 100.0% | 25.0% | 15.65 | 0.66 |
| star_10 | 75.0% | 25.0% | 16.81 | 0.67 |
| star_12 | 75.0% | 25.0% | 19.97 | 0.90 |
| star_6 | 75.0% | 37.5% | 26.52 | 0.56 |

Detailed fault metrics, failure reasons, token counts and timing distributions: [metrics.json](metrics.json).
Requests and observations: requests.jsonl. Actions and ground truth: episodes.jsonl.
One canonical run per physical case: repeat consistency and renamed robustness are not measured here.
The policy consumes current public graph features, not raw full tool transcripts; results apply to this representation.
Fast wrong diagnoses must not be interpreted as an efficiency win without reporting accuracy.
Confidence intervals on this small development sample do not establish broad model superiority.
