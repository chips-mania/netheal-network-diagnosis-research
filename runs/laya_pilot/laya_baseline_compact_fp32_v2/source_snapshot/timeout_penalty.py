"""E2 — On timeout with no diagnosis, apply the original incorrect-diagnosis penalty.

Does not change exact/wrong/partial on diagnose actions.
"""
from __future__ import annotations

import gymnasium as gym

from netheal.environment.actions import ActionCategory

from .action_mask import base_env


def incorrect_diagnosis_penalty(root) -> float:
    calc = root.reward_calculator
    n = len(root.network.get_all_devices()) if root.network else 6
    scale = 1.0 + (n / calc.scaling_factor)
    return float(calc.incorrect_diagnosis_penalty) * scale


class TimeoutPenaltyWrapper(gym.Wrapper):
    timeout_penalty_variant = True

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._saw_diagnosis = False

    def reset(self, **kwargs):
        self._saw_diagnosis = False
        return self.env.reset(**kwargs)

    def step(self, action):
        root = base_env(self.env)
        spec = root.action_space_manager.get_action_spec(int(action))
        if spec is not None and spec.category == ActionCategory.DIAGNOSIS:
            self._saw_diagnosis = True
        obs, reward, terminated, truncated, info = self.env.step(action)
        extra = 0.0
        if truncated and not terminated and not self._saw_diagnosis:
            extra = incorrect_diagnosis_penalty(root)
            reward = float(reward) + extra
        info = dict(info)
        info["timeout_penalty"] = extra
        info["timeout_penalty_applied"] = extra != 0.0
        return obs, reward, terminated, truncated, info


def wrap_with_timeout_penalty(env: gym.Env) -> gym.Env:
    return TimeoutPenaltyWrapper(env)
