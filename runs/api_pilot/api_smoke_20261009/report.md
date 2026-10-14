# GPT-6.1 Sol vs Jev API pilot

This is a development pilot, not the final confirmatory evaluation.
Costs are usage-based USD estimates; invoices have not been reconciled.
Run status: completed; verification: passed.

| Provider | Episodes | Exact | Wrong diagnosis | Mean seconds | p95 seconds | Mean USD | Total USD | Tools |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt | 1 | 100.0% | 0.0% | 26.04 | 26.04 | 0.036068 | 0.036068 | 6.00 |
| jev | 1 | 0.0% | 100.0% | 0.64 | 0.64 | 0.000428 | 0.000428 | 2.00 |

## Per condition

| Condition | GPT Exact | Jev Exact | GPT seconds | Jev seconds |
|---|---:|---:|---:|---:|
| star_6 | 100.0% | 0.0% | 26.04 | 0.64 |

Detailed fault metrics, failure reasons, token counts and timing distributions: [metrics.json](metrics.json).
Requests and observations: requests.jsonl. Actions and ground truth: episodes.jsonl.
One canonical run per physical case: repeat consistency and renamed robustness are not measured here.
The policy consumes current public graph features, not raw full tool transcripts; results apply to this representation.
Fast wrong diagnoses must not be interpreted as an efficiency win without reporting accuracy.
Confidence intervals on this small development sample do not establish broad model superiority.
