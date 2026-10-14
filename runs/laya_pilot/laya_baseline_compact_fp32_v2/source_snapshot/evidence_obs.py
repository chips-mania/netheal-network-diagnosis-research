"""Patch Gym vectors so DOWN / latency are visible. Does not modify NetHeal."""
from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from netheal.environment.observation import ConnectionStatus, DeviceStatus

from .action_mask import base_env

LATENCY_MS_THRESHOLD = 15.0
LATENCY_SCALE = 100.0

STATUS_TO_FEAT = {
    DeviceStatus.UNKNOWN: 0.0,
    DeviceStatus.UP: 1.0,
    DeviceStatus.DOWN: -1.0,
    DeviceStatus.DEGRADED: 0.5,
}


def _status_feat(status: Any) -> float:
    if isinstance(status, DeviceStatus):
        return STATUS_TO_FEAT[status]
    if status is None:
        return 0.0
    name = getattr(status, "name", str(status)).upper()
    if "DOWN" in name:
        return -1.0
    if "DEGRAD" in name:
        return 0.5
    if name in ("UP", "ACTIVE") or "UP" in name:
        return 1.0
    return 0.0


def _tool_latency(result) -> Optional[float]:
    inner = getattr(result, "result", None)
    data = getattr(inner, "data", None) if inner is not None else None
    if not isinstance(data, dict):
        return None
    lat = data.get("latency_ms")
    if lat is None:
        lat = data.get("total_latency_ms")
    if lat is None:
        return None
    return float(lat)


def structured_obs(env):
    root = base_env(env)
    return getattr(root, "observation", None)


def device_row_index(struct, device_id: str) -> int:
    for i, did in enumerate(struct.device_status.devices.keys()):
        if did == device_id:
            return i
    raise KeyError(device_id)


def patch_observation(obs: Dict[str, Any], struct) -> Dict[str, Any]:
    """Overwrite Cisco to_dict holes in-place (same keys/shapes)."""
    if struct is None:
        return obs
    out = dict(obs)
    ds = np.array(obs["device_status"], dtype=np.float32, copy=True)
    disc = np.array(obs["discovery_matrix"], copy=True)
    recent = np.array(obs["recent_diagnostics"], dtype=np.float32, copy=True)

    for i, (_did, info) in enumerate(struct.device_status.devices.items()):
        if i >= ds.shape[0]:
            break
        ds[i, 0] = _status_feat(info.get("status"))

    lat_by_dev: Dict[str, float] = {}
    matrix = struct.discovery_matrix
    for (si, di), props in (matrix.connection_properties or {}).items():
        lat = props.get("latency") if isinstance(props, dict) else None
        if lat is None:
            continue
        lat = float(lat)
        src = matrix.reverse_map.get(si)
        dst = matrix.reverse_map.get(di)
        if src:
            lat_by_dev[src] = max(lat_by_dev.get(src, 0.0), lat)
        if dst:
            lat_by_dev[dst] = max(lat_by_dev.get(dst, 0.0), lat)
        if lat >= LATENCY_MS_THRESHOLD and 0 <= si < disc.shape[0] and 0 <= di < disc.shape[1]:
            disc[si, di] = np.int8(ConnectionStatus.DEGRADED.value)

    results = struct.diagnostic_memory.get_recent_results(recent.shape[0])
    for i, result in enumerate(results):
        lat = _tool_latency(result)
        if lat is not None:
            recent[i, 4] = float(np.clip(lat / LATENCY_SCALE, 0.0, 1.0))
            if result.source:
                lat_by_dev[result.source] = max(lat_by_dev.get(result.source, 0.0), lat)
            if result.destination:
                lat_by_dev[result.destination] = max(lat_by_dev.get(result.destination, 0.0), lat)
        else:
            recent[i, 4] = 0.0

    for i, did in enumerate(struct.device_status.devices.keys()):
        if i >= ds.shape[0]:
            break
        ds[i, 4] = float(np.clip(lat_by_dev.get(did, 0.0) / LATENCY_SCALE, 0.0, 1.0))

    out["device_status"] = ds
    out["discovery_matrix"] = disc
    out["recent_diagnostics"] = recent
    return out


class EvidenceObservationWrapper(gym.ObservationWrapper):
    """Same observation space; status + latency written into existing slots."""

    evidence_obs = True

    def observation(self, observation):
        return patch_observation(observation, structured_obs(self.env))


def wrap_with_evidence_obs(env: gym.Env) -> gym.Env:
    return EvidenceObservationWrapper(env)


def find_evidence_obs(env):
    current = env
    while current is not None:
        if getattr(current, "evidence_obs", False):
            return current
        current = getattr(current, "env", None)
    return None
