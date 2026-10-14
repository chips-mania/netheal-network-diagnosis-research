"""Factorized diagnosis: type commit, then location submit.

Protocol variant F — does not modify the NetHeal package.
Splits the 72-way type×location joint diagnosis into two masked stages.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from netheal.environment.actions import ActionCategory
from netheal.faults.injector import FaultType

from .action_mask import base_env

COMMITTED_TYPE_KEY = "committed_fault_type"
FAULT_TYPE_ORDER = list(FaultType)
DEVICE_LEVEL_TYPES = {FaultType.DEVICE_FAILURE, FaultType.MISCONFIGURATION}


class FactorizedDiagnosisWrapper(gym.Wrapper):
    """Require diagnosis as type (4) then location (6 or 30).

    Phase 1: only one representative diagnosis action per fault type is valid.
    Taking it commits the type, applies a step penalty, and does not terminate.

    Phase 2: only locations for the committed type are valid. Submitting one
    forwards the real diagnosis to NetHeal (exact-match reward / terminate).
    """

    factorized_diagnosis = True

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.committed_type: Optional[FaultType] = None
        self._augment_observation_space()

    def _augment_observation_space(self) -> None:
        base_space = self.env.observation_space
        if not isinstance(base_space, spaces.Dict):
            return
        extra = {
            COMMITTED_TYPE_KEY: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(FAULT_TYPE_ORDER),),
                dtype=np.float32,
            )
        }
        self.observation_space = spaces.Dict({**base_space.spaces, **extra})

    def reset(self, **kwargs):
        self.committed_type = None
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        spec = self._action_spec(action)
        if spec is not None and spec.category == ActionCategory.DIAGNOSIS:
            if int(action) in set(self.get_valid_actions()):
                if self.committed_type is None:
                    return self._commit_type(spec)
                return self._submit_location(int(action), spec)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info

    def get_valid_actions(self) -> List[int]:
        inner = self.env
        if hasattr(inner, "get_valid_actions"):
            valid = list(inner.get_valid_actions())
        else:
            valid = list(base_env(self.env).get_valid_actions())

        root = base_env(self.env)
        if self.committed_type is None:
            reps = set(self._type_representatives(root).values())
            filtered = [
                aid
                for aid in valid
                if not self._is_diagnosis(root, aid) or aid in reps
            ]
            return filtered or valid

        filtered = [
            aid
            for aid in valid
            if not self._is_diagnosis(root, aid)
            or self._diagnosis_type(root, aid) == self.committed_type
        ]
        return filtered or valid

    def _commit_type(self, spec):
        root = base_env(self.env)
        self.committed_type = spec.action_type
        root.step_count += 1
        root.observation.episode_step = root.step_count
        truncated = root.step_count >= root.max_episode_steps
        if truncated:
            root.episode_done = True

        info = {
            "action_spec": spec.to_dict(),
            "factorized_phase": "type_commit",
            "committed_fault_type": self.committed_type.value,
            "reward_breakdown": {"step_penalty": -0.1, "diagnosis_reward": 0.0},
            "valid_actions": self.get_valid_actions(),
        }
        obs = self._augment_obs(root.observation.to_dict())
        return obs, -0.1, False, truncated, info

    def _submit_location(self, action: int, spec):
        root = base_env(self.env)
        submit_id = action
        if spec.action_type != self.committed_type:
            remapped = self._find_diagnosis(
                root, self.committed_type, spec.parameters.get("location", "")
            )
            if remapped is not None:
                submit_id = remapped
        obs, reward, terminated, truncated, info = self.env.step(submit_id)
        info = dict(info)
        info["factorized_phase"] = "location_submit"
        info["committed_fault_type"] = (
            self.committed_type.value if self.committed_type else None
        )
        return self._augment_obs(obs), reward, terminated, truncated, info

    def _augment_obs(self, obs):
        vec = np.zeros(len(FAULT_TYPE_ORDER), dtype=np.float32)
        if self.committed_type is not None:
            vec[FAULT_TYPE_ORDER.index(self.committed_type)] = 1.0
        if isinstance(obs, dict):
            out = dict(obs)
            out[COMMITTED_TYPE_KEY] = vec
            return out
        return obs

    def _type_representatives(self, root) -> Dict[FaultType, int]:
        reps: Dict[FaultType, int] = {}
        for aid, spec in root.action_space_manager.action_map.items():
            if spec.category != ActionCategory.DIAGNOSIS:
                continue
            if spec.action_type not in reps:
                reps[spec.action_type] = aid
        return reps

    def _find_diagnosis(self, root, fault_type: FaultType, location: str) -> Optional[int]:
        for aid, spec in root.action_space_manager.action_map.items():
            if spec.category != ActionCategory.DIAGNOSIS:
                continue
            if spec.action_type == fault_type and spec.parameters.get("location") == location:
                return aid
        return None

    def _action_spec(self, action):
        root = base_env(self.env)
        return root.action_space_manager.get_action_spec(int(action))

    @staticmethod
    def _is_diagnosis(root, action_id: int) -> bool:
        spec = root.action_space_manager.get_action_spec(action_id)
        return spec is not None and spec.category == ActionCategory.DIAGNOSIS

    @staticmethod
    def _diagnosis_type(root, action_id: int):
        spec = root.action_space_manager.get_action_spec(action_id)
        return spec.action_type if spec is not None else None


def wrap_with_factorized_diagnosis(env: gym.Env) -> gym.Env:
    return FactorizedDiagnosisWrapper(env)


def find_factorized_diagnosis(env: gym.Env):
    current = env
    while current is not None:
        if getattr(current, "factorized_diagnosis", False):
            return current
        current = getattr(current, "env", None)
    return None
