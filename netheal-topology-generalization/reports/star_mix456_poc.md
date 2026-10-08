# Mix N=4/5/6 Rule-BC → zero-shot N=8

> Train: Rule demos on Star **N∈{4,5,6}** (pad=12), equal count per N.
> Eval: greedy zero-shot on held-out sizes especially **N=8**.
> Question: multi-size demos vs N=6-only path cloning.

- Demos/N: **700** × {4,5,6} = 2100 eps, transitions=11309
- Per-N collect: {'4': {'ok': 700, 'tries': 700, 'per_fault_ok': {'performance_degradation': 187, 'misconfiguration': 166, 'device_failure': 182, 'link_failure': 165}, 'n_transitions': 3516, 'avg_demo_len': 5.022857142857143, 'n_devices': 4, 'pad': 12, 'max_steps': 15}, '5': {'ok': 700, 'tries': 700, 'per_fault_ok': {'device_failure': 186, 'performance_degradation': 164, 'link_failure': 167, 'misconfiguration': 183}, 'n_transitions': 3842, 'avg_demo_len': 5.488571428571428, 'n_devices': 5, 'pad': 12, 'max_steps': 15}, '6': {'ok': 700, 'tries': 700, 'per_fault_ok': {'device_failure': 151, 'link_failure': 181, 'performance_degradation': 185, 'misconfiguration': 183}, 'n_transitions': 3951, 'avg_demo_len': 5.644285714285714, 'n_devices': 6, 'pad': 12, 'max_steps': 15}}
- Last BC loss: 0.0619
- ckpt: `masked_ppo_bc_rule_pad12_mix456.zip`

## Exact DSR

| N | Rule | Zero-shot mix4/5/6 | Zero-shot N=6-only (prior) | In train? |
|---:|---:|---:|---:|---|
| 4 | 100% | **99%** | 0–22% | yes |
| 5 | 100% | **95%** | — | yes |
| 6 | 100% | **93%** | 97–99% | yes |
| 8 | 98% | **75%** | 76–79% | **held-out** |
| 10 | 99% | **56%** | 54–64% | **held-out** |

## Held-out N=8 detail (mix vs prior)

- mix4/5/6 zero-shot Exact: **75%**
- N=6-only zero-shot Exact (prior): **76–79%**
- Rule ceiling N=8: 98%
- shortcut mix: 0%

- Device 80% (20/25; cons 20)
- Perf 68% (17/25; cons 16)
- Misconfig 92% (23/25; cons 23)
- Link 60% (15/25; cons 0)

## Train-set check N=4/5/6

- N=4: Exact 99%, Device 100% (25/25; cons 25), Perf 96% (24/25; cons 23)
- N=5: Exact 95%, Device 100% (25/25; cons 25), Perf 100% (25/25; cons 22)
- N=6: Exact 93%, Device 100% (25/25; cons 25), Perf 92% (23/25; cons 21)

## Readout

- Held-out **N=8: 75%** vs N=6-only prior **76–79%** → **no gain** (slightly lower).
- Held-out **N=10: 56%** similarly flat vs prior 54–64%.
- Train sizes N=4/5/6 are high (99/95/93%) — especially N=4 recovered vs N=6-only’s ~0%.
- **Verdict:** mixing small sizes teaches those sizes; it does **not** create a size-invariant method that helps unseen larger N. Path-cloning critique stands.

Run: `python -m src.run_star_mix456`
