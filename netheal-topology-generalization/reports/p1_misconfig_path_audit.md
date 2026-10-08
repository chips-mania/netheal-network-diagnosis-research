# Misconfig path audit (N=8 / N=10)

> Fault-only eval. Buckets: A no_evidence · B wrong_hypothesis · C commit_fail · D incomplete_discrimination · success
> Primary model: **Generic**. Misconfig@N10 failures are dominated by **not reaching conclude-match** (D_incomplete_discrimination) — evidence-path / Link–Misconfig discrimination, not commit.

## Bucket rates

| Model | N | Exact | A no_ev | B wrong | C commit | D incomplete |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 8 | 60% | 0% | 0% | 40% | 0% |
| baseline | 10 | 0% | 0% | 0% | 56% | 44% |
| generic | 8 | 56% | 0% | 0% | 10% | 34% |
| generic | 10 | 8% | 0% | 0% | 0% | 92% |

## Cue rates (end of episode)

| Model | N | partial | strong_partial | conclude_match | wrong_conclude | iface@GT | asym_from_GT |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 8 | 100% | 100% | 100% | 0% | 76% | 100% |
| baseline | 10 | 100% | 56% | 56% | 0% | 14% | 100% |
| generic | 8 | 100% | 66% | 66% | 0% | 56% | 100% |
| generic | 10 | 100% | 8% | 8% | 0% | 8% | 100% |

## Samples (Generic N=10)

### D_incomplete_discrimination
- loc=`edge_4` timeout=True submitted=False pred=`None:None`
  - traj: `scan → check_status → check_status → check_interfaces → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status`
  - cues: {'asym_from_gt': True, 'status_gt': None, 'iface_checked_gt': False, 'iface_downs': [], 'n_missing': 1} last_conclude=None
- loc=`edge_8` timeout=True submitted=False pred=`None:None`
  - traj: `scan → check_status → check_status → check_interfaces → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status`
  - cues: {'asym_from_gt': True, 'status_gt': None, 'iface_checked_gt': False, 'iface_downs': [], 'n_missing': 1} last_conclude=None
- loc=`edge_5` timeout=True submitted=False pred=`None:None`
  - traj: `scan → check_status → check_status → check_interfaces → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status → check_status`
  - cues: {'asym_from_gt': True, 'status_gt': None, 'iface_checked_gt': False, 'iface_downs': [], 'n_missing': 1} last_conclude=None
### success
- loc=`center` timeout=False submitted=True pred=`misconfiguration:center`
  - traj: `scan → check_status → check_status → check_interfaces → diagnose`
  - cues: {'asym_from_gt': True, 'status_gt': 'up', 'iface_checked_gt': True, 'iface_downs': ['edge_0'], 'n_missing': 1} last_conclude=None
- loc=`center` timeout=False submitted=True pred=`misconfiguration:center`
  - traj: `scan → check_status → check_status → check_interfaces → diagnose`
  - cues: {'asym_from_gt': True, 'status_gt': 'up', 'iface_checked_gt': True, 'iface_downs': ['edge_0'], 'n_missing': 1} last_conclude=None
- loc=`center` timeout=False submitted=True pred=`misconfiguration:center`
  - traj: `scan → check_status → check_status → check_interfaces → diagnose`
  - cues: {'asym_from_gt': True, 'status_gt': 'up', 'iface_checked_gt': True, 'iface_downs': ['edge_7'], 'n_missing': 1} last_conclude=None

## Readout

Misconfig@N10 failures are dominated by **not reaching conclude-match** (D_incomplete_discrimination) — evidence-path / Link–Misconfig discrimination, not commit.

### Sharper mechanism (Generic N=10)

- **asym_from_GT = 100%** → scan already surfaces the asymmetric missing edge (GT cue present).
- **iface@GT = 8%**, failure samples: `status_gt=None`, `iface_checked_gt=False`, then **status spam**.
- Success samples are almost all GT=`center` with iface checked + downs.
- Spoke Misconfig: policy finds missing but **does not target status/iface on the misconfigured spoke** → Rule never gets enough to conclude → timeout.
- **C_commit_fail = 0%** at Generic N=10 → unlike Link, this is **not** a submit-button problem.

Baseline N=8 still had **40% commit_fail** with conclude_match 100% — Generic moved Misconfig failures from commit toward **incomplete targeting/discrimination**.

Next lever: improve **which node gets status/iface** after asymmetric missing (pointer / curriculum), not more diagnose weight.

Run: `python -m src.run_p1_misconfig_path_audit`
