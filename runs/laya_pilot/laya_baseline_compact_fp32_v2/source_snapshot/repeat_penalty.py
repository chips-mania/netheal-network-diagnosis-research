"""Extra penalty when the same action_id is reused in an episode.

Train-only shaping. Does not modify the NetHeal package.
"""
from __future__ import annotations

from typing import Set

import gymnasium as gym


REPEAT_PENALTY = -2.0  # 20x step cost; fixed before training


class RepeatActionPenaltyWrapper(gym.Wrapper):
    repeat_action_penalty = True

    def __init__(self, env: gym.Env, penalty: float = REPEAT_PENALTY):
        super().__init__(env)
        self.penalty = float(penalty)
        self._seen: Set[int] = set()

    def reset(self, **kwargs):
        self._seen = set()
        return self.env.reset(**kwargs)

    def step(self, action):
        aid = int(action)
        extra = self.penalty if aid in self._seen else 0.0
        self._seen.add(aid)
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward) + extra
        info = dict(info)
        info["repeat_penalty"] = extra
        return obs, reward, terminated, truncated, info


def wrap_with_repeat_penalty(env: gym.Env, penalty: float = REPEAT_PENALTY) -> gym.Env:
    return RepeatActionPenaltyWrapper(env, penalty=penalty)
