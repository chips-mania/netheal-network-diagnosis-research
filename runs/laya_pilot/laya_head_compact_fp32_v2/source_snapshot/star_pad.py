"""Pad Star obs/action to a fixed max_devices so policies transfer across N."""
from __future__ import annotations

from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .action_mask import base_env


def star_canonical_devices(pad_max: int) -> List[str]:
    if pad_max < 2:
        raise ValueError("pad_max must be >= 2")
    return ["center"] + [f"edge_{i}" for i in range(pad_max - 1)]


def steps_budget(n_devices: int, floor: int = 15) -> int:
    """Perf path needs scan + (N-1) spoke pings + diagnose (+ margin)."""
    return max(floor, int(n_devices) + 5)


class StarObsPadWrapper(gym.ObservationWrapper):
    """Zero-pad device_status / discovery_matrix to pad_max (same keys)."""

    star_obs_pad = True

    def __init__(self, env: gym.Env, pad_max: int):
        super().__init__(env)
        self.pad_max = int(pad_max)
        root = base_env(env)
        n0 = int(root.max_devices)
        if self.pad_max < n0:
            raise ValueError(f"pad_max {self.pad_max} < env max_devices {n0}")
        base_space = env.observation_space
        assert isinstance(base_space, spaces.Dict)
        new_spaces = dict(base_space.spaces)
        ds = new_spaces["device_status"]
        disc = new_spaces["discovery_matrix"]
        new_spaces["device_status"] = spaces.Box(
            low=ds.low.min() if hasattr(ds.low, "min") else -np.inf,
            high=ds.high.max() if hasattr(ds.high, "max") else np.inf,
            shape=(self.pad_max, ds.shape[1]),
            dtype=ds.dtype,
        )
        new_spaces["discovery_matrix"] = spaces.Box(
            low=disc.low.min() if hasattr(disc.low, "min") else 0,
            high=disc.high.max() if hasattr(disc.high, "max") else 10,
            shape=(self.pad_max, self.pad_max),
            dtype=disc.dtype,
        )
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, observation):
        out = dict(observation)
        ds = np.array(observation["device_status"], copy=True)
        disc = np.array(observation["discovery_matrix"], copy=True)
        p = self.pad_max
        if ds.shape[0] < p:
            pad_ds = np.zeros((p, ds.shape[1]), dtype=ds.dtype)
            pad_ds[: ds.shape[0]] = ds
            ds = pad_ds
        if disc.shape[0] < p or disc.shape[1] < p:
            pad_disc = np.zeros((p, p), dtype=disc.dtype)
            pad_disc[: disc.shape[0], : disc.shape[1]] = disc
            disc = pad_disc
        out["device_status"] = ds
        out["discovery_matrix"] = disc
        return out


class StarActionPadWrapper(gym.Wrapper):
    """Rebuild action map on canonical center/edge_* so Discrete(n) is fixed across N."""

    star_action_pad = True

    def __init__(self, env: gym.Env, pad_max: int):
        super().__init__(env)
        self.pad_max = int(pad_max)
        self.canonical = star_canonical_devices(self.pad_max)
        root = base_env(env)
        root.action_space_manager.rebuild_for_network(self.canonical)
        self._n = int(root.action_space_manager.total_actions)
        root.action_space = spaces.Discrete(self._n)
        self.action_space = spaces.Discrete(self._n)

    def _sync_action_space(self):
        root = base_env(self)
        root.action_space_manager.rebuild_for_network(self.canonical)
        root.action_space = spaces.Discrete(self._n)
        self.action_space = spaces.Discrete(self._n)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sync_action_space()
        return obs, info

    def step(self, action):
        return self.env.step(action)


def wrap_star_pad(env: gym.Env, pad_max: int) -> gym.Env:
    env = StarObsPadWrapper(env, pad_max)
    env = StarActionPadWrapper(env, pad_max)
    return env


def protocol_for_star_n(protocol: dict, n_devices: int, pad_max: int) -> dict:
    """Shallow copy protocol with min=max=n and step budget; pad applied via wrappers."""
    out = dict(protocol)
    out["min_devices"] = int(n_devices)
    out["max_devices"] = int(n_devices)  # true network size; obs padded separately
    out["max_episode_steps"] = steps_budget(n_devices, floor=int(protocol.get("max_episode_steps", 15)))
    out["_star_pad_max"] = int(pad_max)
    return out
