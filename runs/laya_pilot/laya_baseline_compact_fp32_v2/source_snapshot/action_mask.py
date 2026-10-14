"""Valid-action masking for NetHeal + SB3 MaskablePPO."""
from __future__ import annotations

from typing import Any, List

import gymnasium as gym
import numpy as np

try:
    from sb3_contrib.common.wrappers import ActionMasker
except ImportError:  # pragma: no cover - optional at import time
    ActionMasker = None  # type: ignore


def base_env(env: gym.Env):
    """Reach NetworkTroubleshootingEnv (supports ActionMasker + MetricsCollectorWrapper)."""
    return env.unwrapped


def find_diagnosis_gate(env: gym.Env):
    current = env
    while current is not None:
        if hasattr(current, "min_diagnostic_actions") and hasattr(current, "_diagnostic_count"):
            return current
        current = getattr(current, "env", None)
    return None


def find_evidence_diagnose_gate(env: gym.Env):
    current = env
    while current is not None:
        if getattr(current, "evidence_diagnose_gate", False):
            return current
        current = getattr(current, "env", None)
    return None


def find_factorized_diagnosis(env: gym.Env):
    current = env
    while current is not None:
        if getattr(current, "factorized_diagnosis", False):
            return current
        current = getattr(current, "env", None)
    return None


def valid_action_mask(env: gym.Env) -> np.ndarray:
    """Boolean mask aligned with env.action_space.n from get_valid_actions()."""
    root = base_env(env)
    n = int(root.action_space.n)
    mask = np.zeros(n, dtype=bool)
    factorized = find_factorized_diagnosis(env)
    evi_gate = find_evidence_diagnose_gate(env)
    gate = find_diagnosis_gate(env)
    if factorized is not None:
        valid_ids = factorized.get_valid_actions()
    elif evi_gate is not None:
        valid_ids = evi_gate.get_valid_actions()
    elif gate is not None:
        valid_ids = gate.get_valid_actions()
    else:
        valid_ids = root.get_valid_actions()
    for action_id in valid_ids:
        if 0 <= action_id < n:
            mask[action_id] = True
    if not mask.any() and n > 0:
        # Safety fallback; should not happen in normal NetHeal states.
        mask[0] = True
    return mask


def mask_fn(env: gym.Env) -> np.ndarray:
    return valid_action_mask(env)


def wrap_with_action_mask(env: gym.Env) -> gym.Env:
    if ActionMasker is None:
        raise ImportError(
            "sb3-contrib is required for action masking. Install with: pip install sb3-contrib"
        )
    return ActionMasker(env, mask_fn)


def action_masks_for_env(env: gym.Env) -> np.ndarray:
    if hasattr(env, "action_masks"):
        return env.action_masks()
    inner = getattr(env, "env", None)
    if inner is not None and hasattr(inner, "action_masks"):
        return inner.action_masks()
    return valid_action_mask(env)


def masked_valid_action_ids(env: gym.Env) -> List[int]:
    mask = valid_action_mask(env)
    return [i for i, ok in enumerate(mask) if ok]
