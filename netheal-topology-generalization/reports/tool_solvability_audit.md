# Step 9.10 — Solvability / Tool Information Audit

Purpose: determine whether NetHeal tools expose enough evidence to identify fault type and location,
independent of PPO training.

Network seed: 42, anchor: edge_0

## 9.10-A — Tool information vs healthy baseline (anchor edge_0)

| Fault | Scan | Ping | Status | Interface | Traceroute | Evidence exists? |
|---|:---:|:---:|:---:|:---:|:---:|---|
| device_failure | N | Y | Y | Y | Y | **sufficient** (ping, status, interfaces, traceroute) |
| performance_degradation | N | Y | N | Y | Y | **sufficient** (ping, interfaces, traceroute) |
| misconfiguration | Y | Y | N | Y | Y | **sufficient** (scan, ping, interfaces, traceroute) |

### Fault-type separability at same anchor

- Verdict: **separable**
- Separable pairs: ['device_failure__vs__performance_degradation', 'device_failure__vs__misconfiguration', 'performance_degradation__vs__misconfiguration']

## 9.10-B — Location discriminability (device_failure)

- Locations tested: ['edge_0', 'edge_1', 'edge_2']
- Verdict: **separable**
- Differing pairs: ['edge_0__vs__edge_1', 'edge_0__vs__edge_2', 'edge_1__vs__edge_2']

### Pairwise tool diffs

- `edge_0__vs__edge_1`: ['ping', 'status', 'interfaces', 'traceroute']
- `edge_0__vs__edge_2`: ['ping', 'status', 'interfaces', 'traceroute']
- `edge_1__vs__edge_2`: ['ping', 'status', 'interfaces', 'traceroute']

## 9.10-B — Location discriminability (performance_degradation)

- Locations tested: ['edge_0', 'edge_1', 'edge_2']
- Verdict: **separable**
- Differing pairs: ['center->edge_0__vs__center->edge_1', 'center->edge_0__vs__center->edge_2', 'center->edge_1__vs__center->edge_2']

### Pairwise tool diffs

- `center->edge_0__vs__center->edge_1`: ['ping', 'interfaces', 'traceroute']
- `center->edge_0__vs__center->edge_2`: ['ping', 'interfaces', 'traceroute']
- `center->edge_1__vs__center->edge_2`: ['ping', 'interfaces', 'traceroute']

## 9.10-B — Location discriminability (misconfiguration)

- Locations tested: ['edge_0', 'edge_1', 'edge_2']
- Verdict: **separable**
- Differing pairs: ['edge_0__vs__edge_1', 'edge_0__vs__edge_2', 'edge_1__vs__edge_2']

### Pairwise tool diffs

- `edge_0__vs__edge_1`: ['ping', 'interfaces']
- `edge_0__vs__edge_2`: ['ping', 'interfaces']
- `edge_1__vs__edge_2`: ['ping', 'interfaces']

## Decision guide

### A. Sufficient tool evidence (fault vs healthy)

- Sufficient: ['device_failure', 'performance_degradation', 'misconfiguration']
- Partial: none
- None: none

### B. Location separability

- device_failure: **separable**
- performance_degradation: **separable**
- misconfiguration: **separable**

### Interpretation

- **A + location separable + PPO failed (9.9)** → RL learning problem (100k / curriculum / reward).
- **Partial evidence** → task definition / observation design issue.
- **No evidence** → benchmark limitation; exclude fault from primary RL experiment.
