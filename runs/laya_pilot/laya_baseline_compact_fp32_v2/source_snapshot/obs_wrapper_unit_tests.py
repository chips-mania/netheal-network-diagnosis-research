"""Stage A unit tests for the evidence observation wrapper."""
from __future__ import annotations

import numpy as np

from netheal.environment.actions import DiagnosticAction, TopologyAction
from netheal.environment.observation import ConnectionStatus
from netheal.faults.injector import FaultType

from .action_mask import base_env
from .controlled_faults import inject_fault_at_location
from .env_utils import make_masked_eval_env
from .evidence_obs import device_row_index, structured_obs
from .protocol import load_protocol


def _find(root, pred) -> int:
    for aid, spec in root.action_space_manager.action_map.items():
        if pred(spec):
            return aid
    raise RuntimeError("action not found")


def _scan_then(env, fault: FaultType, location: str, tool_pred):
    env.reset(seed=42)
    root = base_env(env)
    inject_fault_at_location(root, fault, location)
    scan = _find(root, lambda s: s.action_type == TopologyAction.SCAN_NETWORK)
    env.step(scan)
    tool = _find(root, tool_pred)
    obs, _, _, _, _ = env.step(tool)
    return obs, root, structured_obs(env)


def run_obs_wrapper_unit_tests() -> dict:
    protocol = load_protocol()
    tops = ["star"]
    results = {}

    env = make_masked_eval_env(tops, fault_types=[FaultType.DEVICE_FAILURE], protocol=protocol, evidence_obs=True)
    obs, root, struct = _scan_then(
        env,
        FaultType.DEVICE_FAILURE,
        "edge_0",
        lambda s: s.action_type == DiagnosticAction.CHECK_STATUS and s.parameters.get("device") == "edge_0",
    )
    row = device_row_index(struct, "edge_0")
    feat0 = float(obs["device_status"][row, 0])
    results["1_status_down"] = abs(feat0 - (-1.0)) < 1e-5
    env.close()

    env = make_masked_eval_env(tops, fault_types=[FaultType.DEVICE_FAILURE], protocol=protocol, evidence_obs=True)
    obs, root, struct = _scan_then(
        env,
        FaultType.DEVICE_FAILURE,
        "edge_0",
        lambda s: (
            s.action_type == DiagnosticAction.PING
            and s.parameters.get("source") == "edge_1"
            and s.parameters.get("destination") == "edge_0"
        ),
    )
    feat0s = np.asarray(obs["device_status"])[:, 0]
    recent_ok = float(obs["recent_diagnostics"][0, 3]) == 0.0
    results["2_ping_fail"] = (not np.allclose(feat0s, 0.0)) or recent_ok
    if np.allclose(feat0s, 0.0) and not recent_ok:
        results["2_ping_fail"] = False
    env.close()

    env = make_masked_eval_env(
        tops, fault_types=[FaultType.PERFORMANCE_DEGRADATION], protocol=protocol, evidence_obs=True
    )
    obs, root, struct = _scan_then(
        env,
        FaultType.PERFORMANCE_DEGRADATION,
        "center->edge_0",
        lambda s: (
            s.action_type == DiagnosticAction.PING
            and s.parameters.get("source") == "center"
            and s.parameters.get("destination") == "edge_0"
        ),
    )
    degraded = int(ConnectionStatus.DEGRADED.value) in set(np.asarray(obs["discovery_matrix"]).reshape(-1).tolist())
    lat_ds = float(np.max(np.asarray(obs["device_status"])[:, 4]))
    lat_rd = float(np.max(np.asarray(obs["recent_diagnostics"])[:, 4]))
    results["3_perf_latency"] = degraded or lat_ds > 0.15 or lat_rd > 0.15
    env.close()

    env = make_masked_eval_env(tops, fault_types=[FaultType.DEVICE_FAILURE], protocol=protocol, evidence_obs=True)
    obs, root, struct = _scan_then(
        env,
        FaultType.DEVICE_FAILURE,
        "edge_0",
        lambda s: (
            s.action_type == DiagnosticAction.PING
            and s.parameters.get("source") == "edge_1"
            and s.parameters.get("destination") == "edge_2"
        ),
    )
    feat0s = np.asarray(obs["device_status"])[:, 0]
    has_down = bool(np.any(np.isclose(feat0s, -1.0)))
    disc = np.asarray(obs["discovery_matrix"])
    matrix = struct.discovery_matrix
    s1, s2 = matrix.device_map["edge_1"], matrix.device_map["edge_2"]
    pair_degraded = int(disc[s1, s2]) == int(ConnectionStatus.DEGRADED.value)
    results["4_healthy_ping"] = (not has_down) and (not pair_degraded)
    env.close()

    env = make_masked_eval_env(tops, fault_types=[FaultType.DEVICE_FAILURE], protocol=protocol, evidence_obs=False)
    obs, root, struct = _scan_then(
        env,
        FaultType.DEVICE_FAILURE,
        "edge_0",
        lambda s: s.action_type == DiagnosticAction.CHECK_STATUS and s.parameters.get("device") == "edge_0",
    )
    feat0s = np.asarray(obs["device_status"])[:, 0]
    results["5_raw_feat0_zero"] = bool(np.allclose(feat0s, 0.0))
    env.close()

    failed = [k for k, ok in results.items() if not ok]
    if failed:
        raise AssertionError(f"obs wrapper unit tests failed: {failed} ({results})")
    return results
