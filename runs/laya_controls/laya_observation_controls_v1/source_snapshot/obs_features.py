"""Observation flattening and summary helpers for Step 9.8."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


FIELD_NAMES = (
    "discovery_matrix",
    "device_status",
    "recent_diagnostics",
    "episode_metadata",
)


def flatten_obs(obs: Dict[str, Any], fields: Tuple[str, ...] = FIELD_NAMES) -> np.ndarray:
    parts = []
    for name in fields:
        parts.append(np.asarray(obs[name], dtype=np.float32).reshape(-1))
    return np.concatenate(parts)


def obs_field_arrays(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(obs[k]) for k in FIELD_NAMES}


def summarize_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    arrays = obs_field_arrays(obs)
    disc = arrays["discovery_matrix"]
    dev = arrays["device_status"]
    diag = arrays["recent_diagnostics"]
    meta = arrays["episode_metadata"]
    return {
        "discovery_nonzero": int(np.count_nonzero(disc)),
        "discovery_sum": float(disc.sum()),
        "discovery_unique": [int(x) for x in sorted(set(disc.flatten().tolist()))],
        "device_status_nonzero": int(np.count_nonzero(dev)),
        "device_status_sum": float(dev.sum()),
        "recent_diag_nonzero": int(np.count_nonzero(diag)),
        "metadata": meta.tolist(),
    }


def diff_summary(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    b = obs_field_arrays(before)
    a = obs_field_arrays(after)
    disc_delta = a["discovery_matrix"] - b["discovery_matrix"]
    dev_delta = a["device_status"] - b["device_status"]
    return {
        "discovery_changed_cells": int(np.count_nonzero(disc_delta)),
        "discovery_delta_sum": float(disc_delta.sum()),
        "discovery_delta_nonzero_values": [int(x) for x in sorted(set(disc_delta.flatten().tolist())) if x != 0],
        "device_status_changed_cells": int(np.count_nonzero(dev_delta)),
        "device_status_delta_sum": float(dev_delta.sum()),
    }


def pairwise_field_distance(
    obs_a: Dict[str, Any], obs_b: Dict[str, Any], field: str
) -> float:
    a = np.asarray(obs_a[field], dtype=np.float32).reshape(-1)
    b = np.asarray(obs_b[field], dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for {field}")
    return float(np.linalg.norm(a - b))


def pairwise_obs_distance(obs_a: Dict[str, Any], obs_b: Dict[str, Any]) -> Dict[str, float]:
    return {field: pairwise_field_distance(obs_a, obs_b, field) for field in FIELD_NAMES}


def full_vector_distance(obs_a: Dict[str, Any], obs_b: Dict[str, Any]) -> float:
    return float(np.linalg.norm(flatten_obs(obs_a) - flatten_obs(obs_b)))
