# Step 9.11 — Fault-Specific Failure Mode Audit

Purpose: explain **how** each non-link fault fails (type vs location), now that 9.10 ruled out missing evidence.

## Summary

| Fault | PPO failure mode | Root cause hypothesis |
|---|---|---|
| Link | none (DSR 100%) | scan exposes missing edge |
| Device | type mapping | evidence present after tools; policy maps to performance_degradation |
| Performance | location | type learned; scan-only shortcut to fixed edge_1→center |
| Misconfiguration | type discrimination | scan exposes location (missing link); wrong diagnosis category |

---

## 9.11-M — Misconfiguration: why location = 100%?

- Post-scan location probe CV accuracy: **0.0**

### Scan discovery pattern (vs healthy baseline)

| Fault location | Healthy nonzero cells | Fault nonzero cells |
|---|---:|---:|
| `edge_0` | 10 | 9 |
| `edge_1` | 10 | 9 |
| `edge_2` | 10 | 9 |
| `edge_3` | 10 | 9 |
| `edge_4` | 10 | 9 |
| `center` | 10 | 10 |

Misconfiguration scan leaves one fewer discovery cell (blocked port), similar to link failure. If location probe is high, PPO can read fault site from scan-only obs but must still choose fault type.

### PPO wrong-type pattern (from eval trace)

- Wrong-type predictions by diagnosis category: {'performance_degradation': 40, 'link_failure': 38}
- Dominant wrong patterns (actual loc, pred type, pred target):
  - `edge_1|link_failure|edge_1->center`: 23
  - `edge_4|performance_degradation|edge_4->center`: 22
  - `edge_0|performance_degradation|edge_0->center`: 18
  - `center|link_failure|edge_1->center`: 15
- Wrong-type predictions usually preserve location in target (edge_X or edge_X->center), matching scan-visible missing/blocked link pattern.

---

## 9.11-D — Device: why type → performance_degradation?

- Post-scan obs identical for device vs perf at anchor: **True**
- Obs distance after diagnostics (device vs perf): **1.418**
- Type probe after diagnostics (2-class): **None**

### Key tool results (device_failure @ edge_0)

- ping center→edge_0: {'step_index': 14, 'action_label': 'ping', 'target': 'center->edge_0', 'tool_success': False, 'tool_data': {'error': 'Destination device edge_0 is unreachable'}}
- check_status edge_0: {'step_index': 3, 'action_label': 'check_status', 'target': 'edge_0', 'tool_success': True, 'tool_data': {'device_id': 'edge_0', 'status': 'down', 'ip_address': '192.168.1.2', 'device_type': 'switch'}}
- check_interfaces edge_0: {'step_index': 9, 'action_label': 'check_interfaces', 'target': 'edge_0', 'tool_success': False, 'tool_data': {'error': 'Device edge_0 is down'}}

- PPO predicted types: {'performance_degradation': 100}
- Device-only model always predicted performance_degradation despite using ping/interfaces.

If type probe succeeds after diagnostics but PPO predicts performance_degradation, failure is policy mapping (action/reward), not missing evidence.

---

## 9.11-P — Performance: why location guess?

- Post-scan location probe: **0.0** (expected low)
- Post check_interfaces location probe: **0.0**
- PPO fixed guess: **edge_1->center (100% of perf-only eval episodes)**

### Interface latency signatures (center, by faulted link)

- fault `center->edge_0`: interfaces=[{'interface': 'eth-edge_0', 'destination': 'edge_0', 'status': 'up', 'bandwidth_mbps': 282.27902518542805, 'latency_ms': 23.909680907943628}, {'interface': 'eth-edge_1', 'destination': 'edge_1', 'status': 'up', 'bandwidth_mbps': 111.47022440979242, 'latency_ms': 3.8330048510045414}, {'interface': 'eth-edge_2', 'destination': 'edge_2', 'status': 'up', 'bandwidth_mbps': 96.06944430312198, 'latency_ms': 2.398648188583717}, {'interface': 'eth-edge_3', 'destination': 'edge_3', 'status': 'up', 'bandwidth_mbps': 102.75828746297653, 'latency_ms': 1.5469740202583282}, {'interface': 'eth-edge_4', 'destination': 'edge_4', 'status': 'up', 'bandwidth_mbps': 36.27060998702499, 'latency_ms': 1.3947694280899183}]
- fault `center->edge_1`: interfaces=[{'interface': 'eth-edge_0', 'destination': 'edge_0', 'status': 'up', 'bandwidth_mbps': 282.27902518542805, 'latency_ms': 1.5044483216697024}, {'interface': 'eth-edge_1', 'destination': 'edge_1', 'status': 'up', 'bandwidth_mbps': 111.47022440979242, 'latency_ms': 60.916630758314085}, {'interface': 'eth-edge_2', 'destination': 'edge_2', 'status': 'up', 'bandwidth_mbps': 96.06944430312198, 'latency_ms': 2.398648188583717}, {'interface': 'eth-edge_3', 'destination': 'edge_3', 'status': 'up', 'bandwidth_mbps': 102.75828746297653, 'latency_ms': 1.5469740202583282}, {'interface': 'eth-edge_4', 'destination': 'edge_4', 'status': 'up', 'bandwidth_mbps': 36.27060998702499, 'latency_ms': 1.3947694280899183}]
- fault `center->edge_2`: interfaces=[{'interface': 'eth-edge_0', 'destination': 'edge_0', 'status': 'up', 'bandwidth_mbps': 282.27902518542805, 'latency_ms': 1.5044483216697024}, {'interface': 'eth-edge_1', 'destination': 'edge_1', 'status': 'up', 'bandwidth_mbps': 111.47022440979242, 'latency_ms': 3.8330048510045414}, {'interface': 'eth-edge_2', 'destination': 'edge_2', 'status': 'up', 'bandwidth_mbps': 96.06944430312198, 'latency_ms': 38.12089253807211}, {'interface': 'eth-edge_3', 'destination': 'edge_3', 'status': 'up', 'bandwidth_mbps': 102.75828746297653, 'latency_ms': 1.5469740202583282}, {'interface': 'eth-edge_4', 'destination': 'edge_4', 'status': 'up', 'bandwidth_mbps': 36.27060998702499, 'latency_ms': 1.3947694280899183}]

If location probe is low after scan but high after check_interfaces(center), PPO type shortcut without location tools explains 19% DSR.

---

## Implications before 100k / curriculum

1. **Misconfiguration**: teach/limit diagnosis mapping — location signal is scan-like (missing edge).
2. **Device**: reward or action-cost may favor performance_degradation diagnosis despite device-down tools.
3. **Performance**: must incentivize interface/traceroute use; scan→diagnose cannot recover location.

100k is justified only as **targeted** experiments per failure mode, not blind scaling.
