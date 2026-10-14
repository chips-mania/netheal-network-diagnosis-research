"""E1 — Potential-based evidence shaping (training-time oracle).

Does not modify the NetHeal package. Ground truth is used only to compute Φ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, List, Optional, Set, Tuple

import gymnasium as gym

from netheal.environment.actions import ActionCategory
from netheal.faults.injector import FaultType

from .action_mask import base_env

ALPHA = 1.0
GAMMA = 0.99
LATENCY_ABS_MS = 15.0
LATENCY_RATIO = 5.0
ERROR_MARKERS = ("unreachable", "no route to host", "is down")


def _gt(root) -> Tuple[str, str]:
    fault = getattr(root, "ground_truth_fault", None)
    if fault is None:
        return "", ""
    ft = fault.fault_type.value if hasattr(fault.fault_type, "value") else str(fault.fault_type)
    return ft, str(fault.location or "")


def _link_ends(location: str) -> Set[str]:
    if "->" not in location:
        return {location} if location else set()
    return {p.strip() for p in location.split("->", 1) if p.strip()}


def _error_hits(data: dict) -> bool:
    err = str((data or {}).get("error") or "").lower()
    return any(m in err for m in ERROR_MARKERS)


def _anomalous_destinations(pairs: List[Tuple[str, float]]) -> Set[str]:
    if not pairs:
        return set()
    hit: Set[str] = set()
    for dest, lat in pairs:
        if lat >= LATENCY_ABS_MS:
            hit.add(dest)
            continue
        others = [lv for d, lv in pairs if d != dest]
        if others and lat >= LATENCY_RATIO * median(others):
            hit.add(dest)
    return hit


def _tool_payload(action_result) -> Optional[dict]:
    if action_result is None:
        return None
    tool_name = getattr(action_result, "tool_name", None)
    source = getattr(action_result, "source", None)
    dest = getattr(action_result, "destination", None)
    result = getattr(action_result, "result", None)
    data = getattr(result, "data", None) if result is not None else None
    if tool_name is None and isinstance(action_result, dict):
        tool_name = action_result.get("tool_name")
        source = action_result.get("source")
        dest = action_result.get("destination")
        inner = action_result.get("result") or {}
        data = inner.get("data") if isinstance(inner, dict) else getattr(inner, "data", None)
    if not tool_name:
        return None
    return {
        "tool_name": tool_name,
        "source": source,
        "destination": dest,
        "data": data or {},
    }


@dataclass
class EvidenceBits:
    type_bit: int = 0
    loc_bit: int = 0
    events: List[str] = field(default_factory=list)

    @property
    def phi(self) -> int:
        return int(self.type_bit) + int(self.loc_bit)

    def snapshot(self) -> dict:
        return {"type_bit": self.type_bit, "loc_bit": self.loc_bit, "phi": self.phi}


def ingest_tool(bits: EvidenceBits, payload: dict, gt_type: str, gt_loc: str) -> EvidenceBits:
    """Update bits from one tool result. Link GT never rises."""
    if gt_type == FaultType.LINK_FAILURE.value:
        return bits
    tool = payload["tool_name"]
    data = payload.get("data") or {}
    source = payload.get("source")
    dest = payload.get("destination")

    if tool == "scan_network":
        return bits

    if gt_type == FaultType.DEVICE_FAILURE.value:
        if tool == "ping" and _error_hits(data):
            bits.type_bit = 1
            if dest == gt_loc:
                bits.loc_bit = 1
        elif tool == "check_status" and str(data.get("status", "")).lower() == "down":
            bits.type_bit = 1
            device = data.get("device_id") or source
            if device == gt_loc:
                bits.loc_bit = 1

    elif gt_type == FaultType.PERFORMANCE_DEGRADATION.value:
        ends = _link_ends(gt_loc)
        if tool == "check_interfaces":
            ifaces = data.get("interfaces") or []
            pairs = [
                (str(i.get("destination")), float(i.get("latency_ms") or 0.0))
                for i in ifaces
                if i.get("destination") is not None
            ]
            anom = _anomalous_destinations(pairs)
            if anom:
                bits.type_bit = 1
                if anom & ends:
                    bits.loc_bit = 1
        elif tool in ("ping", "traceroute"):
            lat = data.get("latency_ms")
            if lat is None:
                lat = data.get("total_latency_ms")
            if lat is not None and float(lat) >= LATENCY_ABS_MS:
                bits.type_bit = 1
                if dest in ends or source in ends:
                    bits.loc_bit = 1

    elif gt_type == FaultType.MISCONFIGURATION.value:
        ping_err = tool == "ping" and _error_hits(data)
        iface_down = False
        iface_device = source
        if tool == "check_interfaces":
            ifaces = data.get("interfaces") or []
            iface_down = any(str(i.get("status", "")).lower() == "down" for i in ifaces)
        if ping_err or iface_down:
            bits.type_bit = 1
            if ping_err and dest == gt_loc:
                bits.loc_bit = 1
            if iface_down and iface_device == gt_loc:
                bits.loc_bit = 1

    return bits


class PotentialShapingWrapper(gym.Wrapper):
    """r' = r + α(γΦ' − Φ). Observation is unchanged (no GT leak)."""

    pbrs_shaping = True

    def __init__(self, env: gym.Env, *, apply_shaping: bool = True, alpha: float = ALPHA, gamma: float = GAMMA):
        super().__init__(env)
        self.apply_shaping = apply_shaping
        self.alpha = alpha
        self.gamma = gamma
        self.bits = EvidenceBits()
        self.phi_before_diagnosis: Optional[int] = None
        self.bits_before_diagnosis: Optional[dict] = None

    def reset(self, **kwargs):
        self.bits = EvidenceBits()
        self.phi_before_diagnosis = None
        self.bits_before_diagnosis = None
        return self.env.reset(**kwargs)

    def step(self, action):
        root = base_env(self.env)
        gt_type, gt_loc = _gt(root)
        phi_s = 0 if gt_type == FaultType.LINK_FAILURE.value else self.bits.phi
        spec = root.action_space_manager.get_action_spec(int(action))
        is_diagnosis = spec is not None and spec.category == ActionCategory.DIAGNOSIS
        if is_diagnosis and self.phi_before_diagnosis is None:
            self.phi_before_diagnosis = phi_s
            self.bits_before_diagnosis = self.bits.snapshot()

        obs, reward, terminated, truncated, info = self.env.step(action)
        payload = _tool_payload(info.get("action_result"))
        if payload is not None:
            ingest_tool(self.bits, payload, gt_type, gt_loc)

        if terminated or truncated:
            phi_next = 0
        elif gt_type == FaultType.LINK_FAILURE.value:
            phi_next = 0
        else:
            phi_next = self.bits.phi

        shaping = self.alpha * (self.gamma * phi_next - phi_s)
        if self.apply_shaping:
            reward = float(reward) + shaping

        info = dict(info)
        info["pbrs_phi"] = phi_next if not (terminated or truncated) else self.bits.phi
        info["pbrs_phi_used_next"] = phi_next
        info["pbrs_phi_prev"] = phi_s
        info["pbrs_shaping"] = shaping
        info["pbrs_type_bit"] = self.bits.type_bit
        info["pbrs_loc_bit"] = self.bits.loc_bit
        info["pbrs_apply_shaping"] = self.apply_shaping
        return obs, reward, terminated, truncated, info


def wrap_with_pbrs(env: gym.Env, *, apply_shaping: bool = True) -> gym.Env:
    return PotentialShapingWrapper(env, apply_shaping=apply_shaping)


def find_pbrs(env):
    current = env
    while current is not None:
        if getattr(current, "pbrs_shaping", False):
            return current
        current = getattr(current, "env", None)
    return None
