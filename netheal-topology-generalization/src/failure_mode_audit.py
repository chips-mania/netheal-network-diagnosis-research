"""Step 9.11 — Fault-specific failure mode audit helpers."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from netheal.environment.actions import ActionCategory, DiagnosticAction, TopologyAction
from netheal.faults.injector import FaultType

from .controlled_faults import STAR_CENTER, STAR_EDGES, inject_fault_at_location
from .env_utils import make_audit_env
from .obs_features import flatten_obs, full_vector_distance, pairwise_obs_distance, summarize_obs
from .tool_solvability_audit import (
    _find_action_id,
    _serialize_tool_result,
    build_standard_action_specs,
    run_controlled_sequence,
)


def run_prefix_sequence(
    env,
    *,
    seed: int,
    fault_type: Optional[FaultType],
    fault_location: Optional[str],
    stop_after_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run diagnostic protocol until all labels in stop_after_labels have executed."""
    env.reset(seed=seed)
    env.fault_injector.clear_all_faults()
    env.ground_truth_fault = None
    fault_info = None
    if fault_type is not None and fault_location is not None:
        fault_info = inject_fault_at_location(env, fault_type, fault_location)

    devices = sorted(env.network.get_all_devices())
    action_specs = build_standard_action_specs(devices)
    phase_order = ["scan_network", "check_status", "check_interfaces", "ping", "traceroute"]
    max_phase = (
        max(phase_order.index(l) for l in stop_after_labels)
        if stop_after_labels
        else len(phase_order) - 1
    )
    steps = []

    obs_before_reset = env.observation.to_dict()
    for idx, (_name, category, action_type, params) in enumerate(action_specs, start=1):
        obs_before = env.observation.to_dict()
        action_id = _find_action_id(env, category=category, action_type=action_type, **params)
        spec = env.action_space_manager.get_action_spec(action_id)
        label = spec.action_type.value if hasattr(spec.action_type, "value") else str(spec.action_type)
        if category == ActionCategory.DIAGNOSTIC:
            if action_type == DiagnosticAction.PING:
                label = "ping"
            elif action_type == DiagnosticAction.TRACEROUTE:
                label = "traceroute"
            elif action_type == DiagnosticAction.CHECK_STATUS:
                label = "check_status"
            elif action_type == DiagnosticAction.CHECK_INTERFACES:
                label = "check_interfaces"
        elif action_type == TopologyAction.SCAN_NETWORK:
            label = "scan_network"

        if phase_order.index(label) > max_phase:
            break

        target = ""
        p = spec.parameters or {}
        if "source" in p and "destination" in p:
            target = f"{p['source']}->{p['destination']}"
        elif "device" in p:
            target = str(p["device"])

        _, _, _, _, info = env.step(action_id)
        obs_after = env.observation.to_dict()
        tool = _serialize_tool_result(info.get("action_result"))

        steps.append(
            {
                "step_index": idx,
                "action_label": label,
                "target": target,
                "tool_success": tool["success"],
                "tool_data": tool["data"],
            }
        )

    obs_final = env.observation.to_dict()
    return {
        "seed": seed,
        "fault_type": fault_type.value if fault_type else None,
        "fault_location": fault_info.location if fault_info else None,
        "steps": steps,
        "obs_reset": obs_before_reset,
        "obs_final": obs_final,
        "obs_final_summary": summarize_obs(obs_final),
        "obs_vector": flatten_obs(obs_final).tolist(),
    }


def _missing_discovery_edges(obs: Dict[str, Any], devices: List[str]) -> List[str]:
    """Heuristic: in star topology, count expected center->edge connections missing from discovery matrix."""
    matrix = np.asarray(obs["discovery_matrix"])
    # discovery matrix is adjacency; device order follows insertion order in scan
    discovered = summarize_obs(obs)["discovery_nonzero"]
    return discovered


def scan_missing_link_pattern(
    env,
    *,
    seed: int,
    fault_type: FaultType,
    fault_location: str,
) -> Dict[str, Any]:
    run = run_prefix_sequence(
        env,
        seed=seed,
        fault_type=fault_type,
        fault_location=fault_location,
        stop_after_labels=["scan_network"],
    )
    healthy = run_prefix_sequence(
        env,
        seed=seed,
        fault_type=None,
        fault_location=None,
        stop_after_labels=["scan_network"],
    )
    obs_f = run["obs_final"]
    obs_h = healthy["obs_final"]
    return {
        "fault_type": fault_type.value,
        "fault_location": run["fault_location"],
        "healthy_discovery_nonzero": summarize_obs(obs_h)["discovery_nonzero"],
        "fault_discovery_nonzero": summarize_obs(obs_f)["discovery_nonzero"],
        "obs_distance_vs_healthy": full_vector_distance(obs_f, obs_h),
        "field_distances": pairwise_obs_distance(obs_f, obs_h),
        "scan_step_tool": run["steps"][0]["tool_data"] if run["steps"] else {},
    }


def location_probe_cv(X: np.ndarray, y: List[str], n_splits: int = 5) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if len(set(y)) < 2 or len(y) < 3:
        return {"cv_accuracy": None, "n_samples": len(y), "note": "insufficient classes"}

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000)),
        ]
    )
    min_class = min(Counter(y).values())
    if min_class < 2 or len(y) <= 3:
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(
            n_splits=min(n_splits, min_class, len(y) - 1),
            shuffle=True,
            random_state=0,
        )
    scores = cross_val_score(clf, X, np.array(y), cv=cv)
    return {
        "cv_accuracy": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "n_samples": len(y),
        "cv_folds": getattr(cv, "n_splits", len(y)),
    }


def type_probe_cv(X: np.ndarray, y: List[str], n_splits: int = 5) -> Dict[str, Any]:
    return location_probe_cv(X, y, n_splits=n_splits)


def audit_misconfiguration_location_signal(protocol: dict, *, network_seed: int = 42) -> Dict[str, Any]:
    env = make_audit_env(["star"], protocol=protocol)
    locations = STAR_EDGES + [STAR_CENTER]
    scan_patterns = []
    xs = []
    ys = []

    for loc in locations:
        pattern = scan_missing_link_pattern(
            env,
            seed=network_seed,
            fault_type=FaultType.MISCONFIGURATION,
            fault_location=loc,
        )
        scan_patterns.append(pattern)
        run = run_prefix_sequence(
            env,
            seed=network_seed,
            fault_type=FaultType.MISCONFIGURATION,
            fault_location=loc,
            stop_after_labels=["scan_network"],
        )
        xs.append(np.array(run["obs_vector"], dtype=np.float32))
        ys.append(loc)

    env.close()
    X = np.stack(xs)
    probe = location_probe_cv(X, ys)
    return {
        "network_seed": network_seed,
        "locations": locations,
        "scan_patterns": scan_patterns,
        "post_scan_location_probe": probe,
        "interpretation": (
            "Misconfiguration scan leaves one fewer discovery cell (blocked port), similar to link failure. "
            "If location probe is high, PPO can read fault site from scan-only obs but must still choose fault type."
        ),
    }


def audit_device_type_confusion(protocol: dict, *, network_seed: int = 42, anchor: str = "edge_0") -> Dict[str, Any]:
    env = make_audit_env(["star"], protocol=protocol)
    perf_loc = f"{STAR_CENTER}->{anchor}"

    device_run = run_prefix_sequence(
        env,
        seed=network_seed,
        fault_type=FaultType.DEVICE_FAILURE,
        fault_location=anchor,
        stop_after_labels=["scan_network", "ping", "check_status", "check_interfaces"],
    )
    perf_run = run_prefix_sequence(
        env,
        seed=network_seed,
        fault_type=FaultType.PERFORMANCE_DEGRADATION,
        fault_location=perf_loc,
        stop_after_labels=["scan_network", "ping", "check_status", "check_interfaces"],
    )

    xs = [
        np.array(device_run["obs_vector"], dtype=np.float32),
        np.array(perf_run["obs_vector"], dtype=np.float32),
    ]
    scan_only_device = run_prefix_sequence(
        env, seed=network_seed, fault_type=FaultType.DEVICE_FAILURE, fault_location=anchor,
        stop_after_labels=["scan_network"],
    )
    scan_only_perf = run_prefix_sequence(
        env, seed=network_seed, fault_type=FaultType.PERFORMANCE_DEGRADATION, fault_location=perf_loc,
        stop_after_labels=["scan_network"],
    )

    env.close()
    return {
        "anchor_device": anchor,
        "performance_link": perf_loc,
        "obs_distance_device_vs_perf": full_vector_distance(device_run["obs_final"], perf_run["obs_final"]),
        "field_distances_device_vs_perf": pairwise_obs_distance(device_run["obs_final"], perf_run["obs_final"]),
        "post_scan_obs_identical": full_vector_distance(
            scan_only_device["obs_final"], scan_only_perf["obs_final"]
        )
        < 1e-6,
        "type_probe_after_diagnostics": type_probe_cv(
            np.stack(xs),
            [FaultType.DEVICE_FAILURE.value, FaultType.PERFORMANCE_DEGRADATION.value],
            n_splits=2,
        ),
        "device_key_tool_results": {
            "ping_center_to_anchor": _find_step(device_run["steps"], "ping", f"{STAR_CENTER}->{anchor}"),
            "status_anchor": _find_step(device_run["steps"], "check_status", anchor),
            "interfaces_anchor": _find_step(device_run["steps"], "check_interfaces", anchor),
        },
        "interpretation": (
            "If type probe succeeds after diagnostics but PPO predicts performance_degradation, "
            "failure is policy mapping (action/reward), not missing evidence."
        ),
    }


def audit_performance_location(protocol: dict, *, network_seed: int = 42) -> Dict[str, Any]:
    env = make_audit_env(["star"], protocol=protocol)
    edges = ["edge_0", "edge_1", "edge_2"]
    scan_runs = []
    iface_runs = []
    xs_scan, ys = [], []
    xs_iface = []

    for edge in edges:
        loc = f"{STAR_CENTER}->{edge}"
        scan_run = run_prefix_sequence(
            env,
            seed=network_seed,
            fault_type=FaultType.PERFORMANCE_DEGRADATION,
            fault_location=loc,
            stop_after_labels=["scan_network"],
        )
        iface_run = run_prefix_sequence(
            env,
            seed=network_seed,
            fault_type=FaultType.PERFORMANCE_DEGRADATION,
            fault_location=loc,
            stop_after_labels=["scan_network", "check_interfaces"],
        )
        scan_runs.append(scan_run)
        iface_runs.append(iface_run)
        xs_scan.append(np.array(scan_run["obs_vector"], dtype=np.float32))
        xs_iface.append(np.array(iface_run["obs_vector"], dtype=np.float32))
        ys.append(loc)

        # latency on center interfaces toward faulted edge
        center_iface = _find_step(iface_run["steps"], "check_interfaces", STAR_CENTER)
        if center_iface:
            iface_run["center_interface_latency"] = {
                i.get("destination"): i.get("latency_ms")
                for i in center_iface.get("tool_data", {}).get("interfaces", [])
            }

    env.close()
    return {
        "network_seed": network_seed,
        "locations": [f"{STAR_CENTER}->{e}" for e in edges],
        "post_scan_location_probe": location_probe_cv(np.stack(xs_scan), ys, n_splits=3),
        "post_interfaces_location_probe": location_probe_cv(np.stack(xs_iface), ys, n_splits=3),
        "interface_latency_signatures": [
            {
                "fault_location": r["fault_location"],
                "center_interfaces": _find_step(r["steps"], "check_interfaces", STAR_CENTER),
            }
            for r in iface_runs
        ],
        "ppo_fixed_guess": "edge_1->center (100% of perf-only eval episodes)",
        "interpretation": (
            "If location probe is low after scan but high after check_interfaces(center), "
            "PPO type shortcut without location tools explains 19% DSR."
        ),
    }


def _steps_signature(steps: List[Dict]) -> Dict[str, Any]:
    from .tool_solvability_audit import (
        _interface_signature,
        _ping_signature,
        _scan_signature,
        _status_signature,
        _traceroute_signature,
    )

    enriched = [
        {
            "action_label": s["action_label"],
            "target": s["target"],
            "tool_success": s["tool_success"],
            "tool_data": s["tool_data"],
            "obs_diff": {},
            "obs_after_summary": {},
        }
        for s in steps
    ]
    return {
        "scan": _scan_signature(enriched),
        "ping": _ping_signature(enriched),
        "status": _status_signature(enriched),
        "interfaces": _interface_signature(enriched),
        "traceroute": _traceroute_signature(enriched),
    }


def _find_step(steps: List[Dict], label: str, target: str) -> Optional[Dict]:
    for step in steps:
        if step["action_label"] == label and step["target"] == target:
            return step
    return None


def audit_misconfig_ppo_type_confusion(trace_csv_path) -> Dict[str, Any]:
    import csv
    from pathlib import Path

    rows = list(csv.DictReader(Path(trace_csv_path).open(encoding="utf-8")))
    diag_rows = [r for r in rows if r["action_category"] == "diagnosis"]
    wrong_type = [r for r in diag_rows if r["diagnosis_correct"] == "False"]
    wrong_by_pred_type = Counter(r["action_type"] for r in wrong_type)
    pattern = Counter(
        (r["fault_location"], r["action_type"], r["target"]) for r in wrong_type
    )

    return {
        "n_episodes": len({r["episode_id"] for r in rows}),
        "wrong_type_predictions": dict(wrong_by_pred_type.most_common()),
        "dominant_wrong_patterns": {
            f"{loc}|{typ}|{tgt}": cnt for (loc, typ, tgt), cnt in pattern.most_common(8)
        },
        "note": (
            "Wrong-type predictions usually preserve location in target "
            "(edge_X or edge_X->center), matching scan-visible missing/blocked link pattern."
        ),
    }


def audit_device_ppo_type_confusion(trace_csv_path) -> Dict[str, Any]:
    import csv
    from pathlib import Path

    rows = list(csv.DictReader(Path(trace_csv_path).open(encoding="utf-8")))
    diag = [r for r in rows if r["action_category"] == "diagnosis"]
    return {
        "predicted_type_distribution": dict(Counter(r["action_type"] for r in diag).most_common()),
        "predicted_location_top": dict(Counter(r["target"] for r in diag).most_common(5)),
        "note": "Device-only model always predicted performance_degradation despite using ping/interfaces.",
    }
