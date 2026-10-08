"""Unlock DIAGNOSIS only after fault-like evidence appears in the (wrapped) obs.

Not the old Gate (any diagnostic). Healthy dummy ping does not unlock.
Does not modify the NetHeal package. Reward after unlock is original.
"""
from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np

from netheal.environment.actions import ActionCategory, DiagnosticAction

from .action_mask import base_env
from .evidence_obs import LATENCY_MS_THRESHOLD, LATENCY_SCALE

LATENCY_SLOT = LATENCY_MS_THRESHOLD / LATENCY_SCALE  # 0.15


def has_fault_evidence(obs: Dict[str, Any]) -> bool:
    ds = np.asarray(obs["device_status"])
    feat0 = ds[:, 0]
    if np.any(np.isclose(feat0, -1.0)) or np.any(np.isclose(feat0, 0.5)):
        return True
    if ds.shape[1] > 4 and np.any(ds[:, 4] >= LATENCY_SLOT):
        return True
    disc = np.asarray(obs["discovery_matrix"])
    if np.any(disc == 2):
        return True
    recent = np.asarray(obs["recent_diagnostics"])
    used = recent[:, 0] != 0
    failed = recent[:, 3] == 0
    if np.any(used & failed):
        return True
    return False


class EvidenceDiagnoseGateWrapper(gym.Wrapper):
    """Block DIAGNOSIS until has_fault_evidence(obs)."""

    evidence_diagnose_gate = True

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.unlocked = False
        self._last_obs: Dict[str, Any] | None = None

    def reset(self, **kwargs):
        self.unlocked = False
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self.unlocked = has_fault_evidence(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        if not self.unlocked and has_fault_evidence(obs):
            self.unlocked = True
        info = dict(info)
        info["evidence_gate_unlocked"] = self.unlocked
        return obs, reward, terminated, truncated, info

    def get_valid_actions(self):
        root = base_env(self.env)
        valid = list(root.get_valid_actions())
        if self.unlocked:
            return valid
        filtered = []
        for aid in valid:
            spec = root.action_space_manager.get_action_spec(aid)
            if spec is None or spec.category != ActionCategory.DIAGNOSIS:
                filtered.append(aid)
        return filtered or valid


def wrap_with_evidence_diagnose_gate(env: gym.Env) -> gym.Env:
    return EvidenceDiagnoseGateWrapper(env)


def find_evidence_diagnose_gate(env):
    current = env
    while current is not None:
        if getattr(current, "evidence_diagnose_gate", False):
            return current
        current = getattr(current, "env", None)
    return None


def smoke_test() -> dict:
    """Scan must stay locked; failed ping unlocks; healthy ping does not."""
    from netheal.environment.actions import DiagnosticAction, TopologyAction
    from netheal.faults.injector import FaultType

    from .controlled_faults import inject_fault_at_location
    from .env_utils import make_masked_base_env
    from .protocol import load_protocol

    protocol = load_protocol()

    def _setup():
        env = make_masked_base_env(
            ["star"],
            protocol=protocol,
            evidence_obs=True,
            evidence_diagnose_gate=True,
        )
        env.reset(seed=42)
        root = base_env(env)
        inject_fault_at_location(root, FaultType.DEVICE_FAILURE, "edge_0")
        return env, root

    env, root = _setup()
    gate = find_evidence_diagnose_gate(env)
    env.step(_find_action(root, TopologyAction.SCAN_NETWORK))
    locked_after_scan = (not gate.unlocked) and (not _diag_open(env))
    env.step(_find_ping(root, "edge_1", "edge_0"))
    open_after_fail = gate.unlocked and _diag_open(env)
    env.close()

    env, root = _setup()
    env.step(_find_action(root, TopologyAction.SCAN_NETWORK))
    env.step(_find_ping(root, "edge_1", "edge_2"))
    gate = find_evidence_diagnose_gate(env)
    dummy_stays_locked = not gate.unlocked
    env.close()
    return {
        "locked_after_scan": locked_after_scan,
        "open_after_fail_ping": open_after_fail,
        "dummy_healthy_ping_locked": dummy_stays_locked,
    }


def _diag_open(env) -> bool:
    root = base_env(env)
    gate = find_evidence_diagnose_gate(env)
    for aid in gate.get_valid_actions():
        spec = root.action_space_manager.get_action_spec(aid)
        if spec is not None and spec.category == ActionCategory.DIAGNOSIS:
            return True
    return False


def _find_action(root, action_type):
    for aid in root.get_valid_actions():
        spec = root.action_space_manager.get_action_spec(aid)
        if spec is not None and spec.action_type == action_type:
            return aid
    raise RuntimeError(f"missing {action_type}")


def _find_ping(root, src: str, dst: str):
    for aid in root.get_valid_actions():
        spec = root.action_space_manager.get_action_spec(aid)
        if spec is None or spec.action_type != DiagnosticAction.PING:
            continue
        p = spec.parameters or {}
        if p.get("source") == src and p.get("destination") == dst:
            return aid
    raise RuntimeError(f"missing ping {src}->{dst}")
