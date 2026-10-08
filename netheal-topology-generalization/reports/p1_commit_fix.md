# P1 commit fix (diagnose oversample + weighted kind CE)

> Goal: strengthen “evidence ready → diagnose” without GT shortcuts.
> Recipe: warm-start baseline P1; oversample diagnose ×3; kind weight w_diag=4.0, extra ×2.0 on misconfig/link diagnose.
> Checkpoint: `C:/Users/jaewon/Desktop/Project/NW_AI/NW_AI/netheal-topology-generalization/models/checkpoints/graph_pointer_p1_commit.pt`

- Transitions: 10947 → 15147 after oversample; last loss 0.0001

## Exact vs baseline P1

| N | Baseline Exact | Commit Exact | Δ | Baseline TO | Commit TO |
|---:|---:|---:|---:|---:|---:|
| 6 | 100% | 99% | -1% | 0% | 0% |
| 8 | 90% | 96% | +6% | 10% | 4% |
| 10 | 42% | 67% | +25% | 58% | 33% |

## Per-fault (commit P1)

### star_6
- Device 100% (25/25; cons 25)
- Perf 96% (24/25; cons 21)
- Misconfig 100% (25/25; cons 25)
- Link 100% (25/25; cons 0)

### star_8
- Device 100% (25/25; cons 25)
- Perf 96% (24/25; cons 22)
- Misconfig 88% (22/25; cons 22)
- Link 100% (25/25; cons 0)

### star_10
- Device 68% (17/25; cons 17)
- Perf 92% (23/25; cons 22)
- Misconfig 8% (2/25; cons 2)
- Link 100% (25/25; cons 0)

## N=10 commit margin (post-fix)

- Conclude-match: P(diag)=0.9793026367823283 top-1=1.0 top-2=1.0 margin=5.629146722157796
- Timeout final: P(diag)=0.018556063189171256 top-1=0.0 kinds={'check_status': 20, 'ping': 5}
- Timeout∩ready: P(diag)=None top-2=None

### Per-fault conclude-match @ N=10

- device_failure: n=21 P(diag)=0.9663883107049125 top-1=1.0 top-2=1.0 margin=4.165813173566546
- link_failure: n=25 P(diag)=0.9667507338523865 top-1=1.0 top-2=1.0 margin=3.9383960342407227
- misconfiguration: n=5 P(diag)=0.9987807154655457 top-1=1.0 top-2=1.0 margin=6.952186870574951
- performance_degradation: n=24 P(diag)=0.9996196379264196 top-1=1.0 top-2=1.0 margin=8.395128846168518

## Decision: **partial_commit_help**

Run: `python -m src.run_p1_commit_fix`
