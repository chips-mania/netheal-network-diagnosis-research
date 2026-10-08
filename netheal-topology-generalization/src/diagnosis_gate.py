"""Diagnosis gate: require at least one diagnostic tool before any diagnosis.

Protocol variant G — does not modify the NetHeal package.
Leakage-free: uses only executed action categories, not ground truth.
"""
from __future__ import annotations

import gymnasium as gym

from netheal.environment.actions import ActionCategory

from .action_mask import base_env


class DiagnosisGateWrapper(gym.Wrapper):
    """Block DIAGNOSIS until a DIAGNOSTIC action has been taken this episode."""

    def __init__(self, env: gym.Env, *, min_diagnostic_actions: int = 1):
        super().__init__(env)
        self.min_diagnostic_actions = min_diagnostic_actions
        self._diagnostic_count = 0

    def reset(self, **kwargs):
        self._diagnostic_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        spec = self._action_spec(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if spec is not None and spec.category == ActionCategory.DIAGNOSTIC:
            self._diagnostic_count += 1
        return obs, reward, terminated, truncated, info

    def get_valid_actions(self):
        root = base_env(self.env)
        valid = list(root.get_valid_actions())
        if self._diagnostic_count >= self.min_diagnostic_actions:
            return valid
        filtered = []
        for aid in valid:
            spec = root.action_space_manager.get_action_spec(aid)
            if spec is None or spec.category != ActionCategory.DIAGNOSIS:
                filtered.append(aid)
        return filtered or valid

    def _action_spec(self, action):
        root = base_env(self.env)
        return root.action_space_manager.get_action_spec(int(action))


def wrap_with_diagnosis_gate(env: gym.Env, *, min_diagnostic_actions: int = 1) -> gym.Env:
    return DiagnosisGateWrapper(env, min_diagnostic_actions=min_diagnostic_actions)
