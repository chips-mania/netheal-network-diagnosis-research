"""Step 9.6-D: MaskablePPO Star sanity with valid-action masking."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_trace import aggregate_behavior, format_confusion_matrix, run_episode_with_policy, write_trace_csv
from .agents import random_valid_action
from .action_mask import action_masks_for_env
from .env_utils import make_eval_env, make_masked_base_env, make_masked_eval_env, metrics_env
from .metrics_utils import format_summary_row, summarize_evaluator
from .protocol import PROJECT_ROOT, load_protocol


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_masked_ppo(protocol: dict, out_dir: Path) -> MaskablePPO:
    cfg = protocol["ppo_sanity"]
    set_seeds(cfg["policy_seed"])
    env = make_masked_base_env(cfg["topology_types"], protocol=protocol)
    check_env(env, warn=True)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=cfg["policy_seed"],
    )
    model.learn(total_timesteps=cfg["timesteps"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "masked_ppo_sanity_star.zip"
    model.save(str(model_path.with_suffix("")))
    env.close()
    return model


def _make_masked_ppo_selector(model: MaskablePPO):
    def choose(env, obs, _step, _records):
        masks = action_masks_for_env(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        return int(action)

    return choose


def _make_random_valid_selector(rng: random.Random):
    def choose(env, _obs, step, _records):
        max_steps = env.unwrapped.max_episode_steps
        prefer = step >= max_steps
        return random_valid_action(env, rng, prefer_diagnosis=prefer)

    return choose


def assess_go(
    ppo_stats: dict,
    ppo_behavior: dict,
    rv_stats: dict,
    n_episodes: int,
) -> dict:
    dsr_margin = ppo_stats["dsr"] - rv_stats["dsr"]
    min_successes = max(3, int(0.05 * n_episodes))

    cond_dsr = dsr_margin >= 0.05 and (ppo_stats["dsr"] * n_episodes - rv_stats["dsr"] * n_episodes) >= min_successes
    cond_behavior = (
        ppo_behavior["first_action_diagnosis_rate"] < 0.95
        and ppo_stats["topology_coverage"] > 0.0
        and (
            ppo_behavior["scan_usage_rate"] > 0
            or ppo_behavior["ping_usage_rate"] > 0
            or ppo_behavior["check_status_usage_rate"] > 0
            or ppo_behavior["check_interfaces_usage_rate"] > 0
        )
    )
    top_pred = 0
    if ppo_behavior["predicted_diagnosis_distribution"]:
        top_pred = max(ppo_behavior["predicted_diagnosis_distribution"].values()) / n_episodes
    cond_diversity = top_pred < 0.90 and ppo_behavior["invalid_action_rate"] < 0.05

    return {
        "dsr": cond_dsr,
        "behavior": cond_behavior,
        "diversity": cond_diversity,
        "overall_go": cond_dsr and cond_behavior and cond_diversity,
        "dsr_margin": dsr_margin,
        "top_prediction_share": top_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_sanity_star.zip",
    )
    args = parser.parse_args()

    protocol = load_protocol()
    model_dir = PROJECT_ROOT / "models" / "checkpoints"
    reports_dir = PROJECT_ROOT / "reports"
    results_dir = PROJECT_ROOT / "results"

    if args.skip_train:
        model = MaskablePPO.load(str(args.model))
    else:
        print("Training MaskablePPO (Star, masked, 30k)...")
        model = train_masked_ppo(protocol, model_dir)

    cfg = protocol["ppo_sanity"]
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    evaluator_ppo = CompetitionEvaluator()
    evaluator_rv = CompetitionEvaluator()
    ppo_records = []
    ppo_summaries = []
    rng = random.Random(0)

    for i in range(args.eval_episodes):
        fault = faults[i % len(faults)]
        seed = 50000 + i

        env = make_masked_eval_env(cfg["topology_types"], fault_types=[fault], protocol=protocol)
        records, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed,
            agent="masked_ppo",
            choose_action=_make_masked_ppo_selector(model),
        )
        m = metrics_env(env).last_episode_metrics
        if m:
            evaluator_ppo.add_episode_metrics(m)
        ppo_records.extend(records)
        ppo_summaries.append(summary)
        env.close()

        env_rv = make_eval_env(cfg["topology_types"], fault_types=[fault], protocol=protocol)
        records_rv, summary_rv = run_episode_with_policy(
            env_rv,
            episode_id=i,
            seed=seed,
            agent="random_valid",
            choose_action=_make_random_valid_selector(rng),
        )
        m_rv = metrics_env(env_rv).last_episode_metrics
        if m_rv:
            evaluator_rv.add_episode_metrics(m_rv)
        env_rv.close()

    ppo_stats = summarize_evaluator(evaluator_ppo)
    rv_stats = summarize_evaluator(evaluator_rv)
    ppo_behavior = aggregate_behavior(ppo_summaries, ppo_records)
    go = assess_go(ppo_stats, ppo_behavior, rv_stats, args.eval_episodes)

    write_trace_csv(results_dir / "masked_ppo_action_trace.csv", ppo_records)

    lines = [
        "# Step 9.6-D — MaskablePPO Star Sanity",
        "",
        f"Eval episodes: {args.eval_episodes}",
        f"Training: Star only, {protocol['ppo_sanity']['timesteps']} steps, masked valid actions",
        "",
        "## Results",
        "",
        format_summary_row("Random-Valid", rv_stats),
        format_summary_row("Masked PPO", ppo_stats),
        "",
        "## Behavior (Masked PPO)",
        "",
        f"- First-step diagnosis rate: {ppo_behavior['first_action_diagnosis_rate']:.1%}",
        f"- Invalid action rate: {ppo_behavior['invalid_action_rate']:.1%}",
        f"- scan usage: {ppo_behavior['scan_usage_rate']:.1%}",
        f"- ping usage: {ppo_behavior['ping_usage_rate']:.1%}",
        f"- Coverage: {ppo_stats['topology_coverage']:.3f}",
        "",
        "### Fault-type DSR",
        "",
        "| Fault | DSR |",
        "|---|---:|",
    ]
    for ft in protocol["fault_types"]:
        lines.append(f"| {ft} | {ppo_behavior['fault_type_dsr'].get(ft, 0):.1%} |")

    lines.extend(["", "### Confusion matrix", ""])
    lines.extend(format_confusion_matrix(ppo_behavior["confusion_matrix"], protocol["fault_types"]))

    lines.extend(
        [
            "",
            "## GO Assessment (new criteria)",
            "",
            f"- ① DSR > Random-Valid (+5%p): **{'PASS' if go['dsr'] else 'FAIL'}** (margin={go['dsr_margin']:.3f})",
            f"- ② Troubleshooting behavior: **{'PASS' if go['behavior'] else 'FAIL'}**",
            f"- ③ Action diversity (top pred <90%, invalid <5%): **{'PASS' if go['diversity'] else 'FAIL'}**",
            f"- **Overall GO: {'YES' if go['overall_go'] else 'NO'}**",
            "",
            "## Next",
            "",
            "- NO-GO: try 100k masked sanity, link-only curriculum, or observation/reward tuning — not Main Train.",
            "- GO: checkpoint curve (10k/30k/50k/100k) then Step 10.",
        ]
    )

    report_path = reports_dir / "masked_ppo_sanity_gonogo.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for line in lines[:8]:
        print(line)
    print(f"Overall GO: {go['overall_go']}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
