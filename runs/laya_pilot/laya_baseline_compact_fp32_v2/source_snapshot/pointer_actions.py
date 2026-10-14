"""Object-grounded (pointer) actions → NetHeal flat action_id.

Semantic actions do not depend on Discrete index layout across N / topology.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from netheal.environment.actions import ActionCategory, DiagnosticAction, TopologyAction
from netheal.faults.injector import FaultType

from .action_mask import base_env
from .scripted_expert import find_action, find_diagnosis, find_scan


class PointerKind(str, Enum):
    SCAN = "scan"
    CHECK_STATUS = "check_status"
    CHECK_INTERFACES = "check_interfaces"
    PING = "ping"
    DIAGNOSE = "diagnose"


@dataclass(frozen=True)
class PointerAction:
    kind: PointerKind
    node: Optional[str] = None
    src: Optional[str] = None
    dst: Optional[str] = None
    fault_type: Optional[str] = None
    location: Optional[str] = None  # diagnose location string


def pointer_scan() -> PointerAction:
    return PointerAction(kind=PointerKind.SCAN)


def pointer_check_status(node: str) -> PointerAction:
    return PointerAction(kind=PointerKind.CHECK_STATUS, node=str(node))


def pointer_check_interfaces(node: str) -> PointerAction:
    return PointerAction(kind=PointerKind.CHECK_INTERFACES, node=str(node))


def pointer_ping(src: str, dst: str) -> PointerAction:
    return PointerAction(kind=PointerKind.PING, src=str(src), dst=str(dst))


def pointer_diagnose(fault_type: str, location: str) -> PointerAction:
    return PointerAction(
        kind=PointerKind.DIAGNOSE, fault_type=str(fault_type), location=str(location)
    )


def resolve_pointer(env, action: PointerAction) -> int:
    """Map PointerAction → current env Discrete action_id."""
    root = base_env(env)
    k = action.kind
    if k == PointerKind.SCAN:
        return find_scan(root)
    if k == PointerKind.CHECK_STATUS:
        aid = find_action(
            root,
            lambda s, d=action.node: s.action_type == DiagnosticAction.CHECK_STATUS
            and (s.parameters or {}).get("device") == d,
        )
        if aid is None:
            raise RuntimeError(f"check_status missing for {action.node}")
        return aid
    if k == PointerKind.CHECK_INTERFACES:
        aid = find_action(
            root,
            lambda s, d=action.node: s.action_type == DiagnosticAction.CHECK_INTERFACES
            and (s.parameters or {}).get("device") == d,
        )
        if aid is None:
            raise RuntimeError(f"check_interfaces missing for {action.node}")
        return aid
    if k == PointerKind.PING:
        aid = find_action(
            root,
            lambda s, a=action.src, b=action.dst: s.action_type == DiagnosticAction.PING
            and (s.parameters or {}).get("source") == a
            and (s.parameters or {}).get("destination") == b,
        )
        if aid is None:
            raise RuntimeError(f"ping missing {action.src}->{action.dst}")
        return aid
    if k == PointerKind.DIAGNOSE:
        return find_diagnosis(root, action.fault_type or "", action.location or "")
    raise RuntimeError(f"unknown pointer kind {k}")


def encode_flat_as_pointer(env, action_id: int) -> Optional[PointerAction]:
    """Inverse: flat id → pointer (for logging / BC relabel)."""
    root = base_env(env)
    spec = root.action_space_manager.get_action_spec(int(action_id))
    if spec is None:
        return None
    if spec.action_type == TopologyAction.SCAN_NETWORK:
        return pointer_scan()
    if spec.action_type == DiagnosticAction.CHECK_STATUS:
        return pointer_check_status((spec.parameters or {}).get("device") or "")
    if spec.action_type == DiagnosticAction.CHECK_INTERFACES:
        return pointer_check_interfaces((spec.parameters or {}).get("device") or "")
    if spec.action_type == DiagnosticAction.PING:
        p = spec.parameters or {}
        return pointer_ping(p.get("source") or "", p.get("destination") or "")
    if spec.category == ActionCategory.DIAGNOSIS:
        ft = spec.action_type.value if hasattr(spec.action_type, "value") else str(spec.action_type)
        loc = str((spec.parameters or {}).get("location") or "")
        return pointer_diagnose(ft, loc)
    return None


def rule_v2_to_pointer(env, flat_id: int) -> PointerAction:
    p = encode_flat_as_pointer(env, flat_id)
    if p is None:
        raise RuntimeError(f"cannot encode action {flat_id}")
    return p


def smoke_pointer_stable_across_n(protocol, sizes=(4, 6, 8)) -> dict:
    """Same semantic ping/diagnose resolves on each N; flat ids may differ."""
    from netheal.faults.injector import FaultType

    from .env_utils import make_masked_eval_env

    out = {"sizes": {}, "ok": True}
    semantic = None
    for n in sizes:
        env = make_masked_eval_env(
            ["star"],
            fault_types=[FaultType.DEVICE_FAILURE],
            protocol=protocol,
            evidence_obs=True,
            min_devices=n,
            max_devices=n,
            max_episode_steps=20,
        )
        env.reset(seed=42 + n)
        root = base_env(env)
        devices = list(root.network.get_all_devices())
        center = "center" if "center" in devices else devices[0]
        edge = next(d for d in devices if d != center)
        ping_p = pointer_ping(center, edge)
        scan_p = pointer_scan()
        st_p = pointer_check_status(edge)
        ping_id = resolve_pointer(env, ping_p)
        scan_id = resolve_pointer(env, scan_p)
        st_id = resolve_pointer(env, st_p)
        # round-trip
        assert encode_flat_as_pointer(env, ping_id).kind == PointerKind.PING
        assert encode_flat_as_pointer(env, scan_id).kind == PointerKind.SCAN
        row = {
            "n": n,
            "action_space_n": int(root.action_space.n),
            "ping_id": ping_id,
            "scan_id": scan_id,
            "status_id": st_id,
            "ping_endpoints": (center, edge),
        }
        out["sizes"][n] = row
        if semantic is None:
            semantic = (center, edge)
        env.close()
    # flat ping ids should differ across different action_space sizes when N changes
    ids = [out["sizes"][n]["ping_id"] for n in sizes]
    spaces = [out["sizes"][n]["action_space_n"] for n in sizes]
    out["ping_ids"] = ids
    out["action_space_ns"] = spaces
    out["spaces_vary"] = len(set(spaces)) > 1
    out["ok"] = bool(out["spaces_vary"]) and all(i >= 0 for i in ids)
    return out
