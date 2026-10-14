"""GT-using expert: scan → evidence tool → exact diagnosis. Train-only demos."""
from __future__ import annotations

from typing import Optional, Tuple

from netheal.environment.actions import ActionCategory, DiagnosticAction, TopologyAction
from netheal.faults.injector import FaultType

from .action_mask import base_env
from .diagnosis_decomposition import location_match
from .pbrs import _gt


def _spec_type(spec) -> str:
    t = spec.action_type
    return t.value if hasattr(t, "value") else str(t)


def find_action(root, pred) -> Optional[int]:
    for aid, spec in root.action_space_manager.action_map.items():
        if pred(spec):
            return int(aid)
    return None


def find_scan(root) -> int:
    aid = find_action(root, lambda s: s.action_type == TopologyAction.SCAN_NETWORK)
    if aid is None:
        raise RuntimeError("scan missing")
    return aid


def find_evidence_tool(root, gt_type: str, gt_loc: str) -> int:
    devices = list(root.network.get_all_devices()) if root.network else []

    def _ends(loc: str) -> Tuple[str, str]:
        if "->" in loc:
            a, b = loc.split("->", 1)
            return a.strip(), b.strip()
        return "center", loc

    if gt_type == FaultType.DEVICE_FAILURE.value:
        aid = find_action(
            root,
            lambda s: s.action_type == DiagnosticAction.CHECK_STATUS
            and (s.parameters or {}).get("device") == gt_loc,
        )
        if aid is not None:
            return aid
    if gt_type in (FaultType.LINK_FAILURE.value, FaultType.PERFORMANCE_DEGRADATION.value):
        src, dst = _ends(gt_loc)
        aid = find_action(
            root,
            lambda s: s.action_type == DiagnosticAction.PING
            and (s.parameters or {}).get("source") == src
            and (s.parameters or {}).get("destination") == dst,
        )
        if aid is not None:
            return aid
    if gt_type == FaultType.MISCONFIGURATION.value:
        src = "center" if gt_loc != "center" else (devices[0] if devices else "edge_0")
        if src == gt_loc and len(devices) > 1:
            src = next(d for d in devices if d != gt_loc)
        aid = find_action(
            root,
            lambda s: s.action_type == DiagnosticAction.PING
            and (s.parameters or {}).get("source") == src
            and (s.parameters or {}).get("destination") == gt_loc,
        )
        if aid is not None:
            return aid
    raise RuntimeError(f"no evidence tool for {gt_type} {gt_loc}")


def find_diagnosis(root, gt_type: str, gt_loc: str) -> int:
    for aid, spec in root.action_space_manager.action_map.items():
        if spec.category != ActionCategory.DIAGNOSIS:
            continue
        if _spec_type(spec) != gt_type:
            continue
        loc = str((spec.parameters or {}).get("location") or "")
        if location_match(gt_loc, loc, gt_type):
            return int(aid)
    raise RuntimeError(f"no diagnosis for {gt_type} {gt_loc}")


def expert_action(env, step_index: int) -> int:
    root = base_env(env)
    gt_type, gt_loc = _gt(root)
    if step_index == 0:
        return find_scan(root)
    if step_index == 1:
        return find_evidence_tool(root, gt_type, gt_loc)
    return find_diagnosis(root, gt_type, gt_loc)
