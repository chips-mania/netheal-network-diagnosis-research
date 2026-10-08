"""Step 9.5: PPO behavior diagnosis before Main Train."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO

from netheal.environment.actions import ActionCategory, DiagnosticAction, TopologyAction
from netheal.environment.rewards import SparseRewardCalculator
from netheal.faults.injector import FaultType

from .action_trace import (
    aggregate_behavior,
    format_confusion_matrix,
    run_agent_eval,
    write_trace_csv,
)
from .agents import (
    immediate_random_diagnosis_action,
    majority_diagnosis_action,
    random_valid_action,
)
from .env_utils import make_eval_env
from .metrics_utils import format_summary_row
from .protocol import PROJECT_ROOT, load_protocol


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _make_ppo_selector(model: PPO):
    def choose(env, obs, _step, _records):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return choose


def _make_random_valid_selector(rng: random.Random):
    def choose(env, _obs, step, _records):
        max_steps = _base_max_steps(env)
        prefer = step >= max_steps
        return random_valid_action(env, rng, prefer_diagnosis=prefer)

    return choose


def _make_immediate_selector(rng: random.Random):
    def choose(env, _obs, step, _records):
        if step == 1:
            return immediate_random_diagnosis_action(env, rng)
        valid = env.unwrapped.get_valid_actions()
        return valid[0] if valid else 0

    return choose


def _make_majority_selector():
    def choose(env, _obs, step, _records):
        if step == 1:
            return majority_diagnosis_action(env, majority_fault="link_failure")
        valid = env.unwrapped.get_valid_actions()
        return valid[0] if valid else 0

    return choose


def _base_max_steps(env) -> int:
    base = env.unwrapped
    while hasattr(base, "env"):
        base = base.env
    return base.max_episode_steps


def compute_reward_table(network_size: int = 6) -> Dict[str, Any]:
    calc = SparseRewardCalculator()
    scale = 1.0 + (network_size / calc.scaling_factor)
    correct = calc.correct_diagnosis_reward * scale
    wrong = calc.incorrect_diagnosis_penalty * scale
    partial_single = calc.correct_diagnosis_reward * calc.single_device_reward_rate * scale
    partial_multi = calc.correct_diagnosis_reward * calc.partial_device_reward_rate * scale

    explore_then_correct = (network_size - 1) * calc.step_penalty + correct
    immediate_wrong = calc.step_penalty + wrong
    immediate_correct = calc.step_penalty + correct

    return {
        "network_size": network_size,
        "step_penalty": calc.step_penalty,
        "correct_diagnosis_reward_scaled": correct,
        "incorrect_diagnosis_penalty_scaled": wrong,
        "partial_device_reward_single_scaled": partial_single,
        "partial_device_reward_per_device_scaled": partial_multi,
        "invalid_action_penalty": -5.0,
        "diagnosis_terminates_episode": True,
        "max_step_truncation_reward": "no terminal bonus; last step still gets step_penalty",
        "scenarios": {
            "immediate_correct": immediate_correct,
            "immediate_wrong": immediate_wrong,
            "explore_5_steps_then_correct": 5 * calc.step_penalty + correct,
            "explore_10_steps_then_correct": 10 * calc.step_penalty + correct,
            "explore_14_steps_then_correct": 14 * calc.step_penalty + correct,
        },
    }


def observation_diff_report(protocol: dict, seed: int = 42) -> Dict[str, Any]:
    env = make_eval_env(protocol["ppo_sanity"]["topology_types"], protocol=protocol)
    base = env.unwrapped
    while hasattr(base, "env"):
        base = base.env

    obs0, info = env.reset(seed=seed)
    gt = info.get("ground_truth_fault", {})

    def _summarize(obs: dict, label: str) -> Dict[str, Any]:
        disc = np.asarray(obs["discovery_matrix"])
        dev = np.asarray(obs["device_status"])
        diag = np.asarray(obs["recent_diagnostics"])
        meta = np.asarray(obs["episode_metadata"])
        return {
            "label": label,
            "discovery_nonzero": int(np.count_nonzero(disc)),
            "discovery_sum": float(disc.sum()),
            "device_status_nonzero": int(np.count_nonzero(dev)),
            "device_status_sum": float(dev.sum()),
            "recent_diagnostics_nonzero": int(np.count_nonzero(diag)),
            "recent_diagnostics_sum": float(diag.sum()),
            "metadata": meta.tolist(),
        }

    snapshots = [_summarize(obs0, "obs0_reset")]
    steps: List[Dict[str, Any]] = []

    def _find_action(predicate) -> int | None:
        specs = base.action_space_manager.action_map
        for aid, spec in specs.items():
            if predicate(spec):
                return aid
        return None

    action_plan = [
        ("scan_network", lambda s: s.category == ActionCategory.TOPOLOGY_DISCOVERY and s.action_type == TopologyAction.SCAN_NETWORK),
        ("ping", lambda s: s.category == ActionCategory.DIAGNOSTIC and s.action_type == DiagnosticAction.PING),
        ("check_interfaces", lambda s: s.category == ActionCategory.DIAGNOSTIC and s.action_type == DiagnosticAction.CHECK_INTERFACES),
    ]

    obs = obs0
    for name, pred in action_plan:
        aid = _find_action(pred)
        if aid is None:
            steps.append({"action": name, "status": "action_not_found"})
            continue
        obs, reward, terminated, truncated, step_info = env.step(aid)
        snapshots.append(_summarize(obs, f"obs_after_{name}"))
        steps.append(
            {
                "action": name,
                "action_id": aid,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "valid_action": aid in set(base.get_valid_actions()),
            }
        )
        if terminated or truncated:
            break

    deltas = []
    for prev, nxt in zip(snapshots[:-1], snapshots[1:]):
        deltas.append(
            {
                "from": prev["label"],
                "to": nxt["label"],
                "discovery_nonzero_delta": nxt["discovery_nonzero"] - prev["discovery_nonzero"],
                "device_status_nonzero_delta": nxt["device_status_nonzero"] - prev["device_status_nonzero"],
                "recent_diagnostics_nonzero_delta": nxt["recent_diagnostics_nonzero"] - prev["recent_diagnostics_nonzero"],
            }
        )

    env.close()
    return {
        "seed": seed,
        "ground_truth_fault": gt,
        "snapshots": snapshots,
        "steps": steps,
        "deltas": deltas,
        "observation_changes_after_tools": any(
            d["discovery_nonzero_delta"] > 0
            or d["device_status_nonzero_delta"] > 0
            or d["recent_diagnostics_nonzero_delta"] > 0
            for d in deltas
        ),
    }


def assess_go_criteria(
    ppo_stats: Dict[str, Any],
    ppo_behavior: Dict[str, Any],
    random_stats: Dict[str, Any],
    immediate_stats: Dict[str, Any],
) -> Dict[str, Any]:
    cond_perf = (
        ppo_stats["dsr"] > random_stats["dsr"] + 0.05
        and ppo_stats["dsr"] > immediate_stats["dsr"] + 0.05
    )
    cond_behavior = (
        ppo_behavior["first_action_diagnosis_rate"] < 0.80
        and ppo_stats["topology_coverage"] > 0.05
        and ppo_behavior["avg_tool_actions_before_diagnosis"] >= 0.5
    )
    cond_not_blind = abs(ppo_stats["dsr"] - immediate_stats["dsr"]) > 0.02 or cond_behavior

    return {
        "performance": cond_perf,
        "behavior": cond_behavior,
        "not_blind_guess_only": cond_not_blind,
        "overall_go": cond_perf and cond_behavior,
        "recommendation": (
            "Proceed to Star-only 100k sanity ONLY if undertraining suspected; "
            "otherwise fix reward/action-space design first."
            if not (cond_perf and cond_behavior)
            else "Eligible for extended sanity training with checkpoint curve."
        ),
    }


def build_report(
    *,
    n_episodes: int,
    agents: Dict[str, Dict[str, Any]],
    ppo_behavior: Dict[str, Any],
    reward_table: Dict[str, Any],
    obs_diff: Dict[str, Any],
    go: Dict[str, Any],
    protocol: dict,
) -> str:
    lines = [
        "# Step 9.5 — PPO Behavior Diagnosis",
        "",
        f"Eval episodes: {n_episodes} (Star, same seeds as Step 9)",
        "",
        "## 1. Agent Comparison",
        "",
        "| Agent | DSR | AvgSteps | Coverage | First-step Diagnosis | Invalid Action Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, payload in agents.items():
        stats = payload["stats"]
        beh = payload.get("behavior", {})
        lines.append(
            f"| {name} | {_pct(stats['dsr'])} | {stats['avg_steps_all']:.2f} | "
            f"{stats['topology_coverage']:.3f} | "
            f"{_pct(beh.get('first_action_diagnosis_rate', 0.0))} | "
            f"{_pct(beh.get('invalid_action_rate', 0.0))} |"
        )

    lines.extend(
        [
            "",
            "## 2. PPO Action Behavior",
            "",
            f"- First action is diagnosis: **{_pct(ppo_behavior['first_action_diagnosis_rate'])}**",
            f"- Avg tool actions before diagnosis: **{ppo_behavior['avg_tool_actions_before_diagnosis']:.2f}**",
            f"- Invalid action rate: **{_pct(ppo_behavior['invalid_action_rate'])}**",
            f"- scan usage: {_pct(ppo_behavior['scan_usage_rate'])}",
            f"- ping usage: {_pct(ppo_behavior['ping_usage_rate'])}",
            f"- traceroute usage: {_pct(ppo_behavior['traceroute_usage_rate'])}",
            f"- check_status usage: {_pct(ppo_behavior['check_status_usage_rate'])}",
            f"- check_interfaces usage: {_pct(ppo_behavior['check_interfaces_usage_rate'])}",
            "",
            "### Predicted diagnosis distribution (PPO)",
            "",
        ]
    )
    for fault, count in ppo_behavior["predicted_diagnosis_distribution"].items():
        lines.append(f"- {fault}: {count}/{n_episodes} ({_pct(count / n_episodes)})")

    lines.extend(["", "### Actual fault distribution", ""])
    for fault, count in ppo_behavior["actual_fault_distribution"].items():
        lines.append(f"- {fault}: {count}/{n_episodes}")

    lines.extend(["", "### Fault-type DSR (PPO)", "", "| Fault | DSR |", "|---|---:|"])
    for fault in protocol["fault_types"]:
        dsr = ppo_behavior["fault_type_dsr"].get(fault, 0.0)
        lines.append(f"| {fault} | {_pct(dsr)} |")

    lines.extend(["", "### Confusion matrix (actual → predicted)", ""])
    lines.extend(format_confusion_matrix(ppo_behavior["confusion_matrix"], protocol["fault_types"]))

    lines.extend(["", "## 3. Reward Structure", ""])
    lines.append(f"- step_penalty: {reward_table['step_penalty']}")
    lines.append(f"- correct diagnosis (scaled, n=6): {reward_table['correct_diagnosis_reward_scaled']:.2f}")
    lines.append(f"- wrong diagnosis (scaled, n=6): {reward_table['incorrect_diagnosis_penalty_scaled']:.2f}")
    lines.append(f"- invalid action penalty: {reward_table['invalid_action_penalty']}")
    lines.append(f"- diagnosis terminates episode: {reward_table['diagnosis_terminates_episode']}")
    lines.extend(["", "**Scenario totals (network_size=6):**", ""])
    for name, value in reward_table["scenarios"].items():
        lines.append(f"- {name}: {value:.2f}")

    lines.extend(["", "## 4. Observation Changes After Tool Actions", ""])
    lines.append(
        f"- Tools change observation meaningfully: **{obs_diff['observation_changes_after_tools']}**"
    )
    for delta in obs_diff["deltas"]:
        lines.append(
            f"- {delta['from']} → {delta['to']}: "
            f"discovery Δ={delta['discovery_nonzero_delta']}, "
            f"device_status Δ={delta['device_status_nonzero_delta']}, "
            f"diagnostics Δ={delta['recent_diagnostics_nonzero_delta']}"
        )

    lines.extend(["", "## 5. Updated Go/No-Go Assessment", ""])
    lines.append(f"- ① Performance (PPO >> Random & Immediate): **{'PASS' if go['performance'] else 'FAIL'}**")
    lines.append(f"- ② Behavior (troubleshooting sequence): **{'PASS' if go['behavior'] else 'FAIL'}**")
    lines.append(f"- ③ Not blind-guess-only: **{'PASS' if go['not_blind_guess_only'] else 'FAIL'}**")
    lines.append(f"- **Overall Step 9.5 GO: {'YES' if go['overall_go'] else 'NO'}**")
    lines.append(f"- Recommendation: {go['recommendation']}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `results/ppo_action_trace.csv`",
            "- `results/ppo_diagnosis_baselines.json`",
            "- `reports/reward_structure.md`",
            "- `reports/observation_diff.json`",
            "",
            "## Next step",
            "",
            "- If NO-GO: investigate action-space masking / reward before 100k or Main Train.",
            "- If GO: Star-only checkpoint curve (10k/30k/50k/100k) before Step 10.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 9.5 PPO behavior diagnosis")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints" / "ppo_sanity_star.zip",
    )
    args = parser.parse_args()

    protocol = load_protocol()
    results_dir = PROJECT_ROOT / "results"
    reports_dir = PROJECT_ROOT / "reports"

    model = PPO.load(str(args.model))
    rng = random.Random(0)

    print(f"Running PPO diagnosis on {args.episodes} Star episodes...")
    ppo_records, ppo_summaries, ppo_stats = run_agent_eval(
        protocol,
        agent="ppo",
        n_episodes=args.episodes,
        choose_action=_make_ppo_selector(model),
    )
    ppo_behavior = aggregate_behavior(ppo_summaries, ppo_records)
    write_trace_csv(results_dir / "ppo_action_trace.csv", ppo_records)

    print("Running baselines on same seeds...")
    rv_records, rv_summaries, rv_stats = run_agent_eval(
        protocol,
        agent="random_valid",
        n_episodes=args.episodes,
        choose_action=_make_random_valid_selector(rng),
    )
    imm_records, imm_summaries, imm_stats = run_agent_eval(
        protocol,
        agent="immediate_random_diagnosis",
        n_episodes=args.episodes,
        choose_action=_make_immediate_selector(random.Random(0)),
    )
    maj_records, maj_summaries, maj_stats = run_agent_eval(
        protocol,
        agent="majority_diagnosis_link_failure",
        n_episodes=args.episodes,
        choose_action=_make_majority_selector(),
    )

    reward_table = compute_reward_table(network_size=protocol["max_devices"])
    obs_diff = observation_diff_report(protocol)

    go = assess_go_criteria(ppo_stats, ppo_behavior, rv_stats, imm_stats)

    agents = {
        "Random-Valid": {"stats": rv_stats, "behavior": aggregate_behavior(rv_summaries, rv_records)},
        "Immediate-Random-Diagnosis": {
            "stats": imm_stats,
            "behavior": aggregate_behavior(imm_summaries, imm_records),
        },
        "Majority-Diagnosis (link_failure)": {
            "stats": maj_stats,
            "behavior": aggregate_behavior(maj_summaries, maj_records),
        },
        "PPO (30k sanity)": {"stats": ppo_stats, "behavior": ppo_behavior},
    }

    baseline_json = {
        "episodes": args.episodes,
        "agents": {
            name: {
                "stats": payload["stats"],
                "behavior": payload["behavior"],
            }
            for name, payload in agents.items()
        },
        "reward_table": reward_table,
        "go_assessment": go,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "ppo_diagnosis_baselines.json").write_text(
        json.dumps(baseline_json, indent=2), encoding="utf-8"
    )

    reward_md = [
        "# NetHeal Reward Structure (Step 9.5)",
        "",
        "Source: `netheal/environment/rewards.py` + invalid action in `env.py`",
        "",
        json.dumps(reward_table, indent=2),
    ]
    (reports_dir / "reward_structure.md").write_text("\n".join(reward_md) + "\n", encoding="utf-8")
    (reports_dir / "observation_diff.json").write_text(
        json.dumps(obs_diff, indent=2), encoding="utf-8"
    )

    report = build_report(
        n_episodes=args.episodes,
        agents=agents,
        ppo_behavior=ppo_behavior,
        reward_table=reward_table,
        obs_diff=obs_diff,
        go=go,
        protocol=protocol,
    )
    report_path = reports_dir / "ppo_diagnosis_step9_5.md"
    report_path.write_text(report, encoding="utf-8")

    print(format_summary_row("PPO", ppo_stats))
    print(format_summary_row("Random-Valid", rv_stats))
    print(format_summary_row("Immediate-Random-Diagnosis", imm_stats))
    print(format_summary_row("Majority-Diagnosis", maj_stats))
    print(f"First-step diagnosis rate (PPO): {_pct(ppo_behavior['first_action_diagnosis_rate'])}")
    print(f"Step 9.5 overall GO: {go['overall_go']}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
