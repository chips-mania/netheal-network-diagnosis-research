# Step 9.7 — Link-Only Star MaskablePPO Sanity

Config: Star, link_failure only, mask ON, hint OFF, 30k steps, seed 1

Eval episodes: 100

## Results

Random-Valid                    DSR=0.030  TTD-success=15.00  AvgSteps=15.00  Cost=0.280  Cov=0.793  n=100
Masked PPO (link-only)          DSR=1.000  TTD-success=2.00  AvgSteps=2.00  Cost=0.300  Cov=0.700  n=100

- TTD-success (PPO): 2.00
- Avg tool actions before diagnosis: 1.00

## Action behavior

- First-step diagnosis: 0.0%
- scan: 100.0%
- ping: 0.0%
- traceroute: 0.0%
- check_status: 0.0%
- check_interfaces: 0.0%

## Location analysis

### Actual fault locations

- edge_2->center: 15
- edge_0->center: 13
- center->edge_1: 12
- edge_1->center: 10
- center->edge_0: 10
- edge_3->center: 9
- center->edge_4: 9
- center->edge_2: 8
- edge_4->center: 7
- center->edge_3: 7

### Predicted diagnosis locations

- center->edge_2: 23
- center->edge_0: 23
- edge_1->center: 22
- edge_4->center: 16
- center->edge_3: 16

- Unique predicted locations: 5

## Assessment

- Case: **case2_shortcut**
- DSR margin vs Random: 0.970
- Evidence gathering (ping+tools): **False**
- Location diversity (>=3 unique preds): **True**

### Case guide

- **case1_strong**: DSR high, tools before diagnosis → PPO can learn troubleshooting; consider multi-fault next.
- **case2_shortcut**: DSR up but scan→guess → inspect location/reward/observation.
- **case3_fail**: link-only still fails → reward/obs/action representation issue.

## Next

- Do **not** Main Train yet unless case1_strong.
- If promising: try 50k/100k link-only, then Link+Device curriculum.
- If not: reward/observation redesign before multi-fault 100k.
