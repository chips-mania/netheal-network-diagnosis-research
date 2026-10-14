"""Shared helpers for single-fault MaskablePPO sanity runs."""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_trace import EpisodeSummary, StepRecord, aggregate_behavior, run_episode_with_policy
from .action_mask import action_masks_for_env
from .agents import random_valid_action
from .env_utils import make_eval_env, make_masked_base_env, make_masked_eval_env, metrics_env
from .diagnosis_decomposition import aggregate_decomposition, format_matching_table
from .metrics_utils import summarize_evaluator, ttd_success_mean


FAULT_ALIASES = {
    "device_failure": "D",
    "performance_degradation": "P",
    "misconfiguration": "M",
    "link_failure": "L",
}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def model_path_for_fault(fault_name: str, out_dir: Path) -> Path:
    return out_dir / f"masked_ppo_{fault_name}_only_star.zip"


def train_single_fault(
    protocol: dict,
    fault_type: FaultType,
    out_dir: Path,
    *,
    cfg_key: str = "nonlink_single_fault_sanity",
) -> MaskablePPO:
    cfg = protocol[cfg_key]
    set_seeds(cfg["policy_seed"])
    env = make_masked_base_env(cfg["topology_types"], fault_types=[fault_type], protocol=protocol)
    check_env(env, warn=True)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=cfg["policy_seed"],
    )
    model.learn(total_timesteps=cfg["timesteps"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = model_path_for_fault(fault_type.value, out_dir)
    model.save(str(path.with_suffix("")))
    env.close()
    return model


def make_ppo_selector(model: MaskablePPO):
    def choose(env, obs, _step, _records):
        masks = action_masks_for_env(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        return int(action)

    return choose


def make_random_valid_selector(rng: random.Random):
    def choose(env, _obs, step, _records):
        max_steps = metrics_env(env).unwrapped.max_episode_steps
        prefer = step >= max_steps
        return random_valid_action(env, rng, prefer_diagnosis=prefer)

    return choose


def location_analysis(summaries: List[EpisodeSummary]) -> Dict:
    return {
        "actual_location_distribution": dict(Counter(s.fault_location for s in summaries).most_common()),
        "predicted_location_distribution": dict(
            Counter(s.predicted_location for s in summaries if s.predicted_location).most_common()
        ),
        "unique_predicted_locations": len({s.predicted_location for s in summaries if s.predicted_location}),
    }


def format_episode_trace(records: List[StepRecord], episode_id: int) -> str:
    lines = [f"Episode {episode_id}:"]
    for rec in records:
        if rec.episode_id != episode_id:
            continue
        target = f" @ {rec.target}" if rec.target else ""
        lines.append(f"  Step {rec.step} → {rec.action_type}{target} (reward={rec.reward:.1f})")
    return "\n".join(lines)


def sample_success_traces(
    records: List[StepRecord], summaries: List[EpisodeSummary], n: int = 3
) -> List[str]:
    success_eps = [s.episode_id for s in summaries if s.success][:n]
    return [format_episode_trace(records, ep_id) for ep_id in success_eps]


def assess_learnability(ppo_stats: dict, ppo_behavior: dict, rv_stats: dict) -> Dict:
    dsr = ppo_stats["dsr"]
    steps = ppo_stats["avg_steps_all"]
    tool_before = ppo_behavior["avg_tool_actions_before_diagnosis"]
    tool_usage = max(
        ppo_behavior["ping_usage_rate"],
        ppo_behavior["traceroute_usage_rate"],
        ppo_behavior["check_status_usage_rate"],
        ppo_behavior["check_interfaces_usage_rate"],
    )

    if dsr >= 0.50 and (tool_before >= 1.0 or tool_usage >= 0.10 or steps >= 3.0):
        verdict = "valid_troubleshooting_policy"
    elif dsr >= 0.70 and steps <= 2.5:
        verdict = "high_dsr_shortcut"
    elif dsr >= 0.25:
        verdict = "partial_or_lucky"
    else:
        verdict = "no_valid_policy_30k"

    return {
        "verdict": verdict,
        "dsr": dsr,
        "dsr_margin_vs_random": dsr - rv_stats["dsr"],
        "avg_steps": steps,
        "tool_before_diagnosis": tool_before,
        "tool_usage_rate_max": tool_usage,
        "scan_usage": ppo_behavior["scan_usage_rate"],
        "ping_usage": ppo_behavior["ping_usage_rate"],
        "traceroute_usage": ppo_behavior["traceroute_usage_rate"],
        "check_status_usage": ppo_behavior["check_status_usage_rate"],
        "check_interfaces_usage": ppo_behavior["check_interfaces_usage_rate"],
    }


def eval_single_fault_model(
    model: MaskablePPO,
    protocol: dict,
    fault_type: FaultType,
    *,
    n_episodes: int,
    seed_base: int,
    cfg_key: str = "nonlink_single_fault_sanity",
) -> Dict:
    cfg = protocol[cfg_key]
    evaluator_ppo = CompetitionEvaluator()
    evaluator_rv = CompetitionEvaluator()
    records: List[StepRecord] = []
    summaries: List[EpisodeSummary] = []
    rng = random.Random(0)

    for i in range(n_episodes):
        seed = seed_base + i
        env = make_masked_eval_env(cfg["topology_types"], fault_types=[fault_type], protocol=protocol)
        ep_records, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed,
            agent=f"masked_ppo_{fault_type.value}",
            choose_action=make_ppo_selector(model),
        )
        m = metrics_env(env).last_episode_metrics
        if m:
            evaluator_ppo.add_episode_metrics(m)
        records.extend(ep_records)
        summaries.append(summary)
        env.close()

        env_rv = make_eval_env(cfg["topology_types"], fault_types=[fault_type], protocol=protocol)
        run_episode_with_policy(
            env_rv,
            episode_id=i,
            seed=seed,
            agent="random_valid",
            choose_action=make_random_valid_selector(rng),
        )
        m_rv = metrics_env(env_rv).last_episode_metrics
        if m_rv:
            evaluator_rv.add_episode_metrics(m_rv)
        env_rv.close()

    ppo_stats = summarize_evaluator(evaluator_ppo)
    rv_stats = summarize_evaluator(evaluator_rv)
    behavior = aggregate_behavior(summaries, records)
    loc = location_analysis(summaries)
    learn = assess_learnability(ppo_stats, behavior, rv_stats)
    ttd = ttd_success_mean(evaluator_ppo.episodes)
    decomposition = aggregate_decomposition(summaries)

    return {
        "fault_type": fault_type.value,
        "alias": FAULT_ALIASES.get(fault_type.value, "?"),
        "ppo_stats": ppo_stats,
        "rv_stats": rv_stats,
        "behavior": behavior,
        "location_analysis": loc,
        "learnability": learn,
        "ttd_success": ttd,
        "diagnosis_decomposition": decomposition,
        "success_traces": sample_success_traces(records, summaries, n=3),
        "records": records,
    }
