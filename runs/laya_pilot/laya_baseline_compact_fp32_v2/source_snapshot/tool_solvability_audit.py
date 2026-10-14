"""Step 9.10 — Tool information / solvability audit helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from netheal.environment.actions import ActionCategory, DiagnosticAction, TopologyAction
from netheal.faults.injector import FaultType

from .controlled_faults import STAR_CENTER, STAR_EDGES, inject_fault_at_location
from .env_utils import make_audit_env
from .obs_features import diff_summary, flatten_obs, summarize_obs


@dataclass
class AuditStep:
    step_index: int
    action_label: str
    action_id: int
    tool_name: str
    target: str
    tool_success: Optional[bool]
    tool_data: Dict[str, Any]
    obs_before_summary: Dict[str, Any]
    obs_after_summary: Dict[str, Any]
    obs_diff: Dict[str, Any]
    obs_vector_delta_norm: float


def _find_action_id(env, *, category: ActionCategory, action_type, **params) -> int:
    for aid, spec in env.action_space_manager.action_map.items():
        if spec.category != category:
            continue
        if spec.action_type != action_type:
            continue
        if all(spec.parameters.get(k) == v for k, v in params.items()):
            return aid
    raise RuntimeError(f"Action not found: {category}/{action_type}/{params}")


def _serialize_tool_result(action_result) -> Dict[str, Any]:
    if action_result is None:
        return {"tool_name": "", "success": None, "data": {}}
    tr = getattr(action_result, "result", None)
    if tr is None:
        return {"tool_name": getattr(action_result, "tool_name", ""), "success": None, "data": {}}
    return {
        "tool_name": tr.tool_name,
        "success": tr.success,
        "data": dict(tr.data),
    }


def _action_label(spec) -> Tuple[str, str]:
    params = spec.parameters or {}
    if spec.action_type == TopologyAction.SCAN_NETWORK:
        return "scan_network", ""
    if spec.action_type == DiagnosticAction.PING:
        return "ping", f"{params['source']}->{params['destination']}"
    if spec.action_type == DiagnosticAction.TRACEROUTE:
        return "traceroute", f"{params['source']}->{params['destination']}"
    if spec.action_type == DiagnosticAction.CHECK_STATUS:
        return "check_status", str(params["device"])
    if spec.action_type == DiagnosticAction.CHECK_INTERFACES:
        return "check_interfaces", str(params["device"])
    return str(spec.action_type), ""


def build_standard_action_specs(devices: List[str]) -> List[Tuple[str, ActionCategory, Any, Dict[str, Any]]]:
    """Ordered diagnostic protocol: scan → status → interfaces → ping → traceroute."""
    specs: List[Tuple[str, ActionCategory, Any, Dict[str, Any]]] = []
    specs.append(("scan_network", ActionCategory.TOPOLOGY_DISCOVERY, TopologyAction.SCAN_NETWORK, {}))

    for device in sorted(devices):
        specs.append(("check_status", ActionCategory.DIAGNOSTIC, DiagnosticAction.CHECK_STATUS, {"device": device}))

    for device in sorted(devices):
        specs.append(
            ("check_interfaces", ActionCategory.DIAGNOSTIC, DiagnosticAction.CHECK_INTERFACES, {"device": device})
        )

    center = STAR_CENTER if STAR_CENTER in devices else devices[0]
    edges = [d for d in devices if d != center]
    for edge in sorted(edges):
        specs.append(("ping", ActionCategory.DIAGNOSTIC, DiagnosticAction.PING, {"source": center, "destination": edge}))
    for edge in sorted(edges):
        specs.append(("ping", ActionCategory.DIAGNOSTIC, DiagnosticAction.PING, {"source": edge, "destination": center}))
    for edge in sorted(edges):
        specs.append(
            ("traceroute", ActionCategory.DIAGNOSTIC, DiagnosticAction.TRACEROUTE, {"source": center, "destination": edge})
        )
    return specs


def run_controlled_sequence(
    env,
    *,
    seed: int,
    fault_type: Optional[FaultType],
    fault_location: Optional[str],
) -> Dict[str, Any]:
    env.reset(seed=seed)
    env.fault_injector.clear_all_faults()
    env.ground_truth_fault = None
    fault_info = None
    if fault_type is not None and fault_location is not None:
        fault_info = inject_fault_at_location(env, fault_type, fault_location)

    devices = sorted(env.network.get_all_devices())
    action_specs = build_standard_action_specs(devices)
    steps: List[AuditStep] = []
    obs = env.observation.to_dict()

    for idx, (tool_name, category, action_type, params) in enumerate(action_specs, start=1):
        obs_before = env.observation.to_dict()
        action_id = _find_action_id(env, category=category, action_type=action_type, **params)
        spec = env.action_space_manager.get_action_spec(action_id)
        label, target = _action_label(spec)
        _, _, _, _, info = env.step(action_id)
        obs_after = env.observation.to_dict()
        tool = _serialize_tool_result(info.get("action_result"))
        delta = flatten_obs(obs_after) - flatten_obs(obs_before)
        steps.append(
            AuditStep(
                step_index=idx,
                action_label=label,
                action_id=action_id,
                tool_name=tool["tool_name"] or label,
                target=target,
                tool_success=tool["success"],
                tool_data=tool["data"],
                obs_before_summary=summarize_obs(obs_before),
                obs_after_summary=summarize_obs(obs_after),
                obs_diff=diff_summary(obs_before, obs_after),
                obs_vector_delta_norm=float(np.linalg.norm(delta)),
            )
        )
        obs = obs_after

    return {
        "seed": seed,
        "fault_type": fault_type.value if fault_type else None,
        "fault_location": fault_info.location if fault_info else None,
        "devices": devices,
        "steps": [step.__dict__ for step in steps],
    }


def _ping_signature(steps: List[Dict]) -> Dict[str, Any]:
    sig = {}
    for step in steps:
        if step["action_label"] != "ping":
            continue
        sig[step["target"]] = {
            "success": step["tool_success"],
            "latency_ms": step["tool_data"].get("latency_ms"),
            "error": step["tool_data"].get("error"),
        }
    return sig


def _status_signature(steps: List[Dict]) -> Dict[str, str]:
    sig = {}
    for step in steps:
        if step["action_label"] != "check_status":
            continue
        sig[step["target"]] = step["tool_data"].get("status", "unknown")
    return sig


def _interface_signature(steps: List[Dict]) -> Dict[str, Any]:
    sig = {}
    for step in steps:
        if step["action_label"] != "check_interfaces":
            continue
        data = step["tool_data"]
        sig[step["target"]] = {
            "success": step["tool_success"],
            "down_interfaces": data.get("down_interfaces"),
            "interfaces": data.get("interfaces", []),
        }
    return sig


def _traceroute_signature(steps: List[Dict]) -> Dict[str, Any]:
    sig = {}
    for step in steps:
        if step["action_label"] != "traceroute":
            continue
        data = step["tool_data"]
        sig[step["target"]] = {
            "success": step["tool_success"],
            "hops": data.get("hops"),
            "total_latency_ms": data.get("total_latency_ms"),
            "failure_point": data.get("failure_point"),
            "error": data.get("error"),
        }
    return sig


def _scan_signature(steps: List[Dict]) -> Dict[str, Any]:
    for step in steps:
        if step["action_label"] == "scan_network":
            return {
                "discovered_devices": step["tool_data"].get("discovered_devices", []),
                "obs_diff": step["obs_diff"],
                "discovery_nonzero_after": step["obs_after_summary"]["discovery_nonzero"],
            }
    return {}


def extract_signatures(run: Dict[str, Any]) -> Dict[str, Any]:
    steps = run["steps"]
    return {
        "scan": _scan_signature(steps),
        "ping": _ping_signature(steps),
        "status": _status_signature(steps),
        "interfaces": _interface_signature(steps),
        "traceroute": _traceroute_signature(steps),
    }


def _dicts_differ(a: Dict, b: Dict) -> bool:
    return a != b


def assess_tool_discriminability(
    healthy_run: Dict[str, Any],
    fault_run: Dict[str, Any],
) -> Dict[str, Any]:
    healthy = extract_signatures(healthy_run)
    faulty = extract_signatures(fault_run)
    per_tool = {}
    for tool in ("scan", "ping", "status", "interfaces", "traceroute"):
        h = healthy[tool]
        f = faulty[tool]
        differs = _dicts_differ(h, f)
        per_tool[tool] = {
            "differs_from_healthy": differs,
            "healthy_summary": h,
            "fault_summary": f,
        }

    useful = [t for t, v in per_tool.items() if v["differs_from_healthy"]]
    if len(useful) >= 2:
        verdict = "sufficient"
    elif len(useful) == 1:
        verdict = "partial"
    else:
        verdict = "none"

    return {"per_tool": per_tool, "useful_tools": useful, "verdict": verdict}


def assess_location_discriminability(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare signatures across different fault locations for the same fault type."""
    if len(runs) < 2:
        return {"verdict": "insufficient_data", "pairwise_differ": {}}

    sigs = [(r["fault_location"], extract_signatures(r)) for r in runs]
    pairwise = {}
    all_sigs = [s for _, s in sigs]
    for i, (loc_a, sig_a) in enumerate(sigs):
        for j, (loc_b, sig_b) in enumerate(sigs):
            if j <= i:
                continue
            key = f"{loc_a}__vs__{loc_b}"
            tool_diffs = {
                tool: _dicts_differ(sig_a[tool], sig_b[tool]) for tool in ("scan", "ping", "status", "interfaces", "traceroute")
            }
            pairwise[key] = tool_diffs

    # Location separable if full signature differs between at least one pair
    full_diff_pairs = [
        k for k, diffs in pairwise.items() if any(diffs.values())
    ]
    verdict = "separable" if full_diff_pairs else "not_separable"
    return {"verdict": verdict, "pairwise_tool_diffs": pairwise, "differing_pairs": full_diff_pairs}


def assess_fault_type_discriminability(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """At same anchor location, do different fault types produce different tool signatures?"""
    sigs = [(r["fault_type"], extract_signatures(r)) for r in runs]
    pairwise = {}
    for i, (ft_a, sig_a) in enumerate(sigs):
        for j, (ft_b, sig_b) in enumerate(sigs):
            if j <= i:
                continue
            key = f"{ft_a}__vs__{ft_b}"
            pairwise[key] = {
                tool: _dicts_differ(sig_a[tool], sig_b[tool])
                for tool in ("scan", "ping", "status", "interfaces", "traceroute")
            }

    separable_pairs = [k for k, diffs in pairwise.items() if any(diffs.values())]
    verdict = "separable" if separable_pairs else "not_separable"
    return {"verdict": verdict, "pairwise_tool_diffs": pairwise, "separable_pairs": separable_pairs}


def run_fault_type_audit(
    protocol: dict,
    *,
    network_seed: int = 42,
    anchor_location: str = "edge_0",
) -> Dict[str, Any]:
    env = make_audit_env(["star"], protocol=protocol)
    healthy = run_controlled_sequence(env, seed=network_seed, fault_type=None, fault_location=None)

    nonlink = [
        FaultType.DEVICE_FAILURE,
        FaultType.PERFORMANCE_DEGRADATION,
        FaultType.MISCONFIGURATION,
    ]
    per_fault = {}
    fault_runs_for_type_compare = []

    for ft in nonlink:
        if ft == FaultType.PERFORMANCE_DEGRADATION:
            loc = f"{STAR_CENTER}->{anchor_location}"
        else:
            loc = anchor_location
        run = run_controlled_sequence(env, seed=network_seed, fault_type=ft, fault_location=loc)
        disc = assess_tool_discriminability(healthy, run)
        per_fault[ft.value] = {
            "fault_location": run["fault_location"],
            "discriminability": disc,
            "signatures": extract_signatures(run),
        }
        fault_runs_for_type_compare.append(run)

    env.close()
    return {
        "network_seed": network_seed,
        "anchor_location": anchor_location,
        "healthy_signatures": extract_signatures(healthy),
        "per_fault": per_fault,
        "fault_type_discriminability": assess_fault_type_discriminability(fault_runs_for_type_compare),
    }


def run_location_audit(
    protocol: dict,
    fault_type: FaultType,
    *,
    network_seed: int = 42,
    locations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from .controlled_faults import default_probe_locations

    env = make_audit_env(["star"], protocol=protocol)
    locs = locations or default_probe_locations(fault_type)
    runs = []
    for loc in locs:
        if fault_type == FaultType.PERFORMANCE_DEGRADATION:
            inject_loc = f"{STAR_CENTER}->{loc}"
        elif fault_type == FaultType.LINK_FAILURE:
            inject_loc = f"{STAR_CENTER}->{loc}"
        else:
            inject_loc = loc
        runs.append(
            run_controlled_sequence(
                env,
                seed=network_seed,
                fault_type=fault_type,
                fault_location=inject_loc,
            )
        )
    env.close()
    return {
        "fault_type": fault_type.value,
        "network_seed": network_seed,
        "locations": locs,
        "runs": runs,
        "location_discriminability": assess_location_discriminability(runs),
        "signatures_by_location": {r["fault_location"]: extract_signatures(r) for r in runs},
    }
