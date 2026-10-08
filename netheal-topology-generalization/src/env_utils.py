"""NetHeal environment factory."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from netheal import NetworkTroubleshootingEnv
from netheal.evaluation.wrapper import MetricsCollectorWrapper
from netheal.faults.injector import FaultType

from .action_mask import base_env

from .protocol import load_protocol


def apply_reward_mode(env, reward_mode: str = "original") -> None:
    """Swap env reward calculator (protocol variant, does not modify NetHeal package)."""
    root = base_env(env)
    scaling = root.reward_calculator.scaling_factor
    if reward_mode == "symmetric":
        from .symmetric_reward import SymmetricPartialRewardCalculator

        root.reward_calculator = SymmetricPartialRewardCalculator(scaling_factor=scaling)
    elif reward_mode != "original":
        raise ValueError(f"Unknown reward_mode: {reward_mode}")


def make_base_env(
    topology_types: List[str],
    fault_types: Optional[List[FaultType]] = None,
    protocol: Optional[Dict[str, Any]] = None,
    *,
    max_episode_steps: Optional[int] = None,
    min_devices: Optional[int] = None,
    max_devices: Optional[int] = None,
    reward_mode: str = "original",
) -> NetworkTroubleshootingEnv:
    cfg = protocol or load_protocol()
    env = NetworkTroubleshootingEnv(
        min_devices=min_devices if min_devices is not None else cfg["min_devices"],
        max_devices=max_devices if max_devices is not None else cfg["max_devices"],
        max_episode_steps=max_episode_steps or cfg["max_episode_steps"],
        topology_types=topology_types,
        fault_types=fault_types or [FaultType(ft) for ft in cfg["fault_types"]],
        enable_user_hints=cfg["enable_user_hints"],
    )
    apply_reward_mode(env, reward_mode)
    return env


def make_audit_env(
    topology_types: List[str],
    protocol: Optional[Dict[str, Any]] = None,
) -> NetworkTroubleshootingEnv:
    """Enough steps for full scan + status + interfaces + ping + traceroute sweep."""
    return make_base_env(topology_types, protocol=protocol, max_episode_steps=40)


def make_eval_env(
    topology_types: List[str],
    fault_types: Optional[List[FaultType]] = None,
    protocol: Optional[Dict[str, Any]] = None,
    *,
    reward_mode: str = "original",
) -> MetricsCollectorWrapper:
    base = make_base_env(
        topology_types, fault_types=fault_types, protocol=protocol, reward_mode=reward_mode
    )
    return MetricsCollectorWrapper(base)


def make_masked_base_env(
    topology_types: List[str],
    fault_types: Optional[List[FaultType]] = None,
    protocol: Optional[Dict[str, Any]] = None,
    *,
    reward_mode: str = "original",
    diagnosis_gate: bool = False,
    factorized_diagnosis: bool = False,
    pbrs: bool = False,
    pbrs_apply_shaping: bool = True,
    timeout_penalty: bool = False,
    evidence_obs: bool = False,
    evidence_diagnose_gate: bool = False,
    repeat_penalty: bool = False,
    episode_log_path: Optional[Any] = None,
    min_devices: Optional[int] = None,
    max_devices: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    star_pad_max: Optional[int] = None,
):
    from .action_mask import wrap_with_action_mask
    from .diagnosis_gate import wrap_with_diagnosis_gate
    from .evidence_diagnose_gate import wrap_with_evidence_diagnose_gate
    from .evidence_obs import wrap_with_evidence_obs
    from .factorized_diagnosis import wrap_with_factorized_diagnosis
    from .pbrs import wrap_with_pbrs
    from .repeat_penalty import wrap_with_repeat_penalty
    from .star_pad import wrap_star_pad
    from .timeout_penalty import wrap_with_timeout_penalty
    from .train_episode_log import wrap_with_train_episode_log

    env = make_base_env(
        topology_types,
        fault_types=fault_types,
        protocol=protocol,
        reward_mode=reward_mode,
        min_devices=min_devices,
        max_devices=max_devices,
        max_episode_steps=max_episode_steps,
    )
    if star_pad_max is not None:
        env = wrap_star_pad(env, star_pad_max)
    if evidence_obs:
        env = wrap_with_evidence_obs(env)
    if episode_log_path is not None:
        env = wrap_with_train_episode_log(env, Path(episode_log_path))
    if evidence_diagnose_gate:
        env = wrap_with_evidence_diagnose_gate(env)
    if diagnosis_gate:
        env = wrap_with_diagnosis_gate(env)
    if factorized_diagnosis:
        env = wrap_with_factorized_diagnosis(env)
    if pbrs:
        env = wrap_with_pbrs(env, apply_shaping=pbrs_apply_shaping)
    if timeout_penalty:
        env = wrap_with_timeout_penalty(env)
    if repeat_penalty:
        env = wrap_with_repeat_penalty(env)
    return wrap_with_action_mask(env)


def make_masked_eval_env(
    topology_types: List[str],
    fault_types: Optional[List[FaultType]] = None,
    protocol: Optional[Dict[str, Any]] = None,
    *,
    reward_mode: str = "original",
    diagnosis_gate: bool = False,
    factorized_diagnosis: bool = False,
    pbrs: bool = False,
    pbrs_apply_shaping: bool = False,
    timeout_penalty: bool = False,
    evidence_obs: bool = False,
    evidence_diagnose_gate: bool = False,
    min_devices: Optional[int] = None,
    max_devices: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    star_pad_max: Optional[int] = None,
):
    """ActionMasker(Metrics(Timeout?(PBRS?(Factorized?(Gate?(Obs?(base))))))).

    Eval DSR uses original success. Timeout wrapper is train-only unless requested.
    """
    from .action_mask import wrap_with_action_mask
    from .diagnosis_gate import wrap_with_diagnosis_gate
    from .evidence_diagnose_gate import wrap_with_evidence_diagnose_gate
    from .evidence_obs import wrap_with_evidence_obs
    from .factorized_diagnosis import wrap_with_factorized_diagnosis
    from .pbrs import wrap_with_pbrs
    from .star_pad import wrap_star_pad
    from .timeout_penalty import wrap_with_timeout_penalty

    base = make_base_env(
        topology_types,
        fault_types=fault_types,
        protocol=protocol,
        reward_mode=reward_mode,
        min_devices=min_devices,
        max_devices=max_devices,
        max_episode_steps=max_episode_steps,
    )
    if star_pad_max is not None:
        base = wrap_star_pad(base, star_pad_max)
    if evidence_obs:
        base = wrap_with_evidence_obs(base)
    if evidence_diagnose_gate:
        base = wrap_with_evidence_diagnose_gate(base)
    if diagnosis_gate:
        base = wrap_with_diagnosis_gate(base)
    if factorized_diagnosis:
        base = wrap_with_factorized_diagnosis(base)
    if pbrs:
        base = wrap_with_pbrs(base, apply_shaping=pbrs_apply_shaping)
    if timeout_penalty:
        base = wrap_with_timeout_penalty(base)
    return wrap_with_action_mask(MetricsCollectorWrapper(base))


def metrics_env(env) -> MetricsCollectorWrapper:
    """Find MetricsCollectorWrapper inside ActionMasker/other wrappers."""
    current = env
    while current is not None:
        if hasattr(current, "last_episode_metrics"):
            return current
        current = getattr(current, "env", None)
    return env
