"""Baseline agents."""
from __future__ import annotations

import random
from typing import List, Optional

import gymnasium as gym

from netheal.environment.actions import ActionCategory


def get_valid_action_ids(env: gym.Env) -> List[int]:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    return list(base.get_valid_actions())


def random_valid_action(
    env: gym.Env,
    rng: random.Random,
    *,
    prefer_diagnosis: bool = False,
) -> int:
    valid = get_valid_action_ids(env)
    if not valid:
        raise RuntimeError("No valid actions available")

    base = env.unwrapped if hasattr(env, "unwrapped") else env
    while hasattr(base, "env"):
        base = base.env
    specs = base.action_space_manager.action_map

    if prefer_diagnosis:
        diagnosis = [a for a in valid if specs[a].category == ActionCategory.DIAGNOSIS]
        if diagnosis:
            return rng.choice(diagnosis)

    non_diagnosis = [a for a in valid if specs[a].category != ActionCategory.DIAGNOSIS]
    pool = non_diagnosis or valid
    return rng.choice(pool)


def run_random_valid_episode(
    env: gym.Env,
    seed: int,
    *,
    force_diagnosis_last: bool = True,
) -> None:
    rng = random.Random(seed)
    env.reset(seed=seed)
    max_steps = env.unwrapped.max_episode_steps

    for step in range(max_steps):
        prefer = force_diagnosis_last and step >= max_steps - 1
        action = random_valid_action(env, rng, prefer_diagnosis=prefer)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break


def _base_env(env: gym.Env):
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    while hasattr(base, "env"):
        base = base.env
    return base


def all_diagnosis_action_ids(env: gym.Env) -> List[int]:
    """All diagnosis actions in the full action map (not validity-filtered)."""
    base = _base_env(env)
    specs = base.action_space_manager.action_map
    return [aid for aid, spec in specs.items() if spec.category == ActionCategory.DIAGNOSIS]


def immediate_random_diagnosis_action(env: gym.Env, rng: random.Random) -> int:
    """Step-1 blind guess from full diagnosis action space (PPO-like, no masking)."""
    pool = all_diagnosis_action_ids(env)
    if not pool:
        raise RuntimeError("No diagnosis actions in action map")
    return rng.choice(pool)


def majority_diagnosis_action(env: gym.Env, majority_fault: str = "link_failure") -> int:
    """Always pick the first diagnosis action for the majority fault type."""
    base = _base_env(env)
    specs = base.action_space_manager.action_map
    for aid, spec in sorted(specs.items()):
        if spec.category != ActionCategory.DIAGNOSIS:
            continue
        fault = spec.action_type.value if hasattr(spec.action_type, "value") else str(spec.action_type)
        if fault == majority_fault:
            return aid
    pool = all_diagnosis_action_ids(env)
    return pool[0]
