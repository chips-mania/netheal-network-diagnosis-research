"""Step 9.7: Link-only Star MaskablePPO diagnostic sanity."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_trace import aggregate_behavior, run_episode_with_policy, write_trace_csv
from .action_mask import action_masks_for_env
from .agents import random_valid_action
from .env_utils import make_eval_env, make_masked_base_env, make_masked_eval_env, metrics_env
from .metrics_utils import format_summary_row, summarize_evaluator, ttd_success_mean
from .protocol import PROJECT_ROOT, load_protocol


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_link_only(protocol: dict, out_dir: Path) -> MaskablePPO:
    cfg = protocol["link_only_sanity"]
    set_seeds(cfg["policy_seed"])
    faults = [FaultType(ft) for ft in cfg["fault_types"]]
    env = make_masked_base_env(cfg["topology_types"], fault_types=faults, protocol=protocol)
    check_env(env, warn=True)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=cfg["policy_seed"],
    )
    model.learn(total_timesteps=cfg["timesteps"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "masked_ppo_link_only_star.zip"
    model.save(str(path.with_suffix("")))
    env.close()
    return model


def _make_ppo_selector(model: MaskablePPO):
    def choose(env, obs, _step, _records):
        masks = action_masks_for_env(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        return int(action)

    return choose


def _make_random_valid_selector(rng: random.Random):
    def choose(env, _obs, step, _records):
        max_steps = metrics_env(env).unwrapped.max_episode_steps
        prefer = step >= max_steps
        return random_valid_action(env, rng, prefer_diagnosis=prefer)

    return choose


def link_location_analysis(summaries) -> Dict:
    actual = Counter(s.fault_location for s in summaries)
    predicted = Counter(s.predicted_location for s in summaries if s.predicted_location)
    success_pairs = [
        (s.fault_location, s.predicted_location)
        for s in summaries
        if s.success and s.predicted_location
    ]
    return {
        "actual_location_distribution": dict(actual.most_common()),
        "predicted_location_distribution": dict(predicted.most_common()),
        "unique_predicted_locations": len(predicted),
        "successful_location_matches": len(success_pairs),
    }


def assess_link_only(
    ppo_stats: dict,
    ppo_behavior: dict,
    rv_stats: dict,
    loc_analysis: dict,
    n_episodes: int,
) -> dict:
    tool_before = ppo_behavior["avg_tool_actions_before_diagnosis"]
    non_diag = sum(
        v
        for k, v in ppo_behavior.get("action_type_frequency", {}).items()
        if k not in ("", "diagnosis")
    )
    case = "case3_fail"
    if ppo_stats["dsr"] >= 0.50 and tool_before >= 1.5 and ppo_behavior["ping_usage_rate"] > 0.1:
        case = "case1_strong"
    elif ppo_stats["dsr"] >= 0.20 and ppo_stats["avg_steps_all"] <= 3.0:
        case = "case2_shortcut"
    elif ppo_stats["dsr"] < 0.20:
        case = "case3_fail"

    return {
        "case": case,
        "dsr_pass": ppo_stats["dsr"] >= rv_stats["dsr"] + 0.10,
        "evidence_gathering": tool_before >= 1.0 and ppo_behavior["ping_usage_rate"] > 0,
        "location_diversity": loc_analysis["unique_predicted_locations"] >= 3,
        "overall_promising": case == "case1_strong",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_link_only_star.zip",
    )
    args = parser.parse_args()

    protocol = load_protocol()
    cfg = protocol["link_only_sanity"]
    faults = [FaultType(ft) for ft in cfg["fault_types"]]
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"

    if args.skip_train:
        model = MaskablePPO.load(str(args.model))
    else:
        print("Training MaskablePPO link-only (Star, 30k)...")
        model = train_link_only(protocol, PROJECT_ROOT / "models" / "checkpoints")

    evaluator_ppo = CompetitionEvaluator()
    evaluator_rv = CompetitionEvaluator()
    ppo_records: List = []
    ppo_summaries: List = []
    rng = random.Random(0)

    for i in range(args.eval_episodes):
        seed = 70000 + i
        env = make_masked_eval_env(cfg["topology_types"], fault_types=faults, protocol=protocol)
        records, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed,
            agent="masked_ppo_link_only",
            choose_action=_make_ppo_selector(model),
        )
        m = metrics_env(env).last_episode_metrics
        if m:
            evaluator_ppo.add_episode_metrics(m)
        ppo_records.extend(records)
        ppo_summaries.append(summary)
        env.close()

        env_rv = make_eval_env(cfg["topology_types"], fault_types=faults, protocol=protocol)
        _, summary_rv = run_episode_with_policy(
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
    loc_analysis = link_location_analysis(ppo_summaries)
    assessment = assess_link_only(ppo_stats, ppo_behavior, rv_stats, loc_analysis, args.eval_episodes)

    write_trace_csv(results / "link_only_action_trace.csv", ppo_records)

    ttd = ttd_success_mean(evaluator_ppo.episodes)
    lines = [
        "# Step 9.7 — Link-Only Star MaskablePPO Sanity",
        "",
        "Config: Star, link_failure only, mask ON, hint OFF, 30k steps, seed 1",
        "",
        f"Eval episodes: {args.eval_episodes}",
        "",
        "## Results",
        "",
        format_summary_row("Random-Valid", rv_stats),
        format_summary_row("Masked PPO (link-only)", ppo_stats),
        "",
        f"- TTD-success (PPO): {ttd:.2f}" if ttd else "- TTD-success (PPO): n/a",
        f"- Avg tool actions before diagnosis: {ppo_behavior['avg_tool_actions_before_diagnosis']:.2f}",
        "",
        "## Action behavior",
        "",
        f"- First-step diagnosis: {ppo_behavior['first_action_diagnosis_rate']:.1%}",
        f"- scan: {ppo_behavior['scan_usage_rate']:.1%}",
        f"- ping: {ppo_behavior['ping_usage_rate']:.1%}",
        f"- traceroute: {ppo_behavior['traceroute_usage_rate']:.1%}",
        f"- check_status: {ppo_behavior['check_status_usage_rate']:.1%}",
        f"- check_interfaces: {ppo_behavior['check_interfaces_usage_rate']:.1%}",
        "",
        "## Location analysis",
        "",
        "### Actual fault locations",
        "",
    ]
    for loc, cnt in loc_analysis["actual_location_distribution"].items():
        lines.append(f"- {loc}: {cnt}")
    lines.extend(["", "### Predicted diagnosis locations", ""])
    for loc, cnt in loc_analysis["predicted_location_distribution"].items():
        lines.append(f"- {loc}: {cnt}")

    lines.extend(
        [
            "",
            f"- Unique predicted locations: {loc_analysis['unique_predicted_locations']}",
            "",
            "## Assessment",
            "",
            f"- Case: **{assessment['case']}**",
            f"- DSR margin vs Random: {ppo_stats['dsr'] - rv_stats['dsr']:.3f}",
            f"- Evidence gathering (ping+tools): **{assessment['evidence_gathering']}**",
            f"- Location diversity (>=3 unique preds): **{assessment['location_diversity']}**",
            "",
            "### Case guide",
            "",
            "- **case1_strong**: DSR high, tools before diagnosis → PPO can learn troubleshooting; consider multi-fault next.",
            "- **case2_shortcut**: DSR up but scan→guess → inspect location/reward/observation.",
            "- **case3_fail**: link-only still fails → reward/obs/action representation issue.",
            "",
            "## Next",
            "",
            "- Do **not** Main Train yet unless case1_strong.",
            "- If promising: try 50k/100k link-only, then Link+Device curriculum.",
            "- If not: reward/observation redesign before multi-fault 100k.",
        ]
    )

    report_path = reports / "link_only_sanity.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (results / "link_only_sanity_summary.json").write_text(
        json.dumps(
            {
                "ppo_stats": ppo_stats,
                "rv_stats": rv_stats,
                "ppo_behavior": ppo_behavior,
                "loc_analysis": loc_analysis,
                "assessment": assessment,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for line in lines[6:12]:
        print(line)
    print(f"Case: {assessment['case']}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
