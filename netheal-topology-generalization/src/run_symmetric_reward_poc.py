"""Step 9.13-A — Symmetric partial reward PoC vs original reward control."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_mask import action_masks_for_env
from .action_trace import aggregate_behavior, run_episode_with_policy, write_trace_csv
from .diagnosis_decomposition import aggregate_decomposition
from .env_utils import make_masked_base_env, make_masked_eval_env, metrics_env
from .metrics_utils import summarize_evaluator
from .protocol import PROJECT_ROOT, load_protocol


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_model(
    protocol: dict,
    *,
    reward_mode: str,
    out_path: Path,
) -> MaskablePPO:
    cfg = protocol["symmetric_reward_poc"]
    set_seeds(cfg["policy_seed"])
    env = make_masked_base_env(
        cfg["topology_types"],
        protocol=protocol,
        reward_mode=reward_mode,
    )
    check_env(env, warn=True)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=cfg["policy_seed"],
    )
    model.learn(total_timesteps=cfg["timesteps"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path.with_suffix("")))
    env.close()
    return model


def _make_ppo_selector(model: MaskablePPO):
    def choose(env, obs, _step, _records):
        masks = action_masks_for_env(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        return int(action)

    return choose


def evaluate_model(
    model: MaskablePPO,
    protocol: dict,
    *,
    n_episodes: int,
    seed_base: int,
    agent_label: str,
) -> Dict:
    """Evaluate policy; DSR uses original NetHeal success criterion (eval env reward=original)."""
    cfg = protocol["symmetric_reward_poc"]
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    evaluator = CompetitionEvaluator()
    records: List = []
    summaries: List = []

    for i in range(n_episodes):
        fault = faults[i % len(faults)]
        seed = seed_base + i
        env = make_masked_eval_env(
            cfg["topology_types"],
            fault_types=[fault],
            protocol=protocol,
            reward_mode="original",
        )
        ep_records, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed,
            agent=agent_label,
            choose_action=_make_ppo_selector(model),
        )
        m = metrics_env(env).last_episode_metrics
        if m:
            evaluator.add_episode_metrics(m)
        records.extend(ep_records)
        summaries.append(summary)
        env.close()

    stats = summarize_evaluator(evaluator)
    behavior = aggregate_behavior(summaries, records)
    decomp = aggregate_decomposition(summaries)
    per_fault_dsr = behavior.get("fault_type_dsr", {})

    return {
        "stats": stats,
        "behavior": behavior,
        "decomposition": {
            "fault_type_accuracy": decomp["fault_type_accuracy"],
            "location_accuracy": decomp["location_accuracy"],
            "exact_diagnosis_accuracy": decomp["exact_diagnosis_accuracy"],
        },
        "per_fault_dsr": per_fault_dsr,
        "records": records,
    }


def assess_poc(control: Dict, variant: Dict) -> Dict:
    c_dsr = control["decomposition"]["exact_diagnosis_accuracy"]
    v_dsr = variant["decomposition"]["exact_diagnosis_accuracy"]
    delta = v_dsr - c_dsr

    c_type = control["decomposition"]["fault_type_accuracy"]
    v_type = variant["decomposition"]["fault_type_accuracy"]
    c_loc = control["decomposition"]["location_accuracy"]
    v_loc = variant["decomposition"]["location_accuracy"]

    complementary_reduced = (
        abs(v_type - v_loc) < abs(c_type - c_loc)
        or (min(v_type, v_loc) > min(c_type, c_loc))
    )

    improved_faults = sum(
        1
        for ft in control["per_fault_dsr"]
        if variant["per_fault_dsr"].get(ft, 0) > control["per_fault_dsr"].get(ft, 0) + 0.05
    )

    top_pred_share = 0.0
    if variant["behavior"]["predicted_diagnosis_distribution"]:
        n = variant["stats"]["episodes"]
        top_pred_share = max(variant["behavior"]["predicted_diagnosis_distribution"].values()) / max(1, n)

    success = (
        delta >= 0.15
        and v_dsr >= 0.35
        and complementary_reduced
        and improved_faults >= 2
        and top_pred_share < 0.90
    )

    return {
        "dsr_delta": delta,
        "complementary_reduced": complementary_reduced,
        "nonlink_faults_improved": improved_faults,
        "top_prediction_share": top_pred_share,
        "verdict": "promising" if success else "inconclusive_or_fail",
        "next_step": (
            "Consider masked multi-fault 100k with symmetric reward (protocol variant)."
            if success
            else "Proceed to 9.13-B Diagnosis Gate ablation; do not scale to 100k yet."
        ),
    }


def build_report(control: Dict, variant: Dict, assessment: Dict, cfg: dict) -> str:
    c, v = control, variant
    lines = [
        "# Step 9.13-A — Symmetric Partial Reward PoC",
        "",
        "> **Protocol variant R** — symmetric type/location partial reward applied uniformly to all faults.",
        "> Control uses original NetHeal `SparseRewardCalculator`.",
        "",
        f"Train: Star, 6 devices, 4 faults, MaskablePPO, {cfg['timesteps']} steps, seed {cfg['policy_seed']}",
        f"Eval: {cfg.get('eval_episodes', 100)} episodes, seed base {cfg.get('eval_seed_base', 95000)}",
        "",
        "## Comparison",
        "",
        "| Metric | Control (Original) | Variant R (Symmetric) |",
        "|---|---:|---:|",
        f"| Exact DSR | {c['decomposition']['exact_diagnosis_accuracy']:.0%} | "
        f"{v['decomposition']['exact_diagnosis_accuracy']:.0%} |",
        f"| Type Accuracy | {c['decomposition']['fault_type_accuracy']:.0%} | "
        f"{v['decomposition']['fault_type_accuracy']:.0%} |",
        f"| Location Accuracy | {c['decomposition']['location_accuracy']:.0%} | "
        f"{v['decomposition']['location_accuracy']:.0%} |",
        f"| AvgSteps | {c['stats']['avg_steps_all']:.1f} | {v['stats']['avg_steps_all']:.1f} |",
        f"| Tools before dx | {c['behavior']['avg_tool_actions_before_diagnosis']:.1f} | "
        f"{v['behavior']['avg_tool_actions_before_diagnosis']:.1f} |",
        f"| Coverage | {c['stats']['topology_coverage']:.2f} | {v['stats']['topology_coverage']:.2f} |",
        "",
        "### Per-fault DSR",
        "",
        "| Fault | Control | Variant R |",
        "|---|---:|---:|",
    ]
    for ft in ("link_failure", "device_failure", "performance_degradation", "misconfiguration"):
        lines.append(
            f"| {ft} | {c['per_fault_dsr'].get(ft, 0):.0%} | {v['per_fault_dsr'].get(ft, 0):.0%} |"
        )

    lines.extend(
        [
            "",
            "## Complementary failure check",
            "",
            f"- Control: Type {c['decomposition']['fault_type_accuracy']:.0%} / "
            f"Loc {c['decomposition']['location_accuracy']:.0%}",
            f"- Variant: Type {v['decomposition']['fault_type_accuracy']:.0%} / "
            f"Loc {v['decomposition']['location_accuracy']:.0%}",
            "",
            "## Assessment",
            "",
            f"- DSR delta: **{assessment['dsr_delta']:+.0%}**",
            f"- Complementary gap reduced: **{assessment['complementary_reduced']}**",
            f"- Non-link faults improved (≥5%p): **{assessment['nonlink_faults_improved']}/3**",
            f"- Top single prediction share: **{assessment['top_prediction_share']:.0%}**",
            f"- Verdict: **{assessment['verdict']}**",
            f"- Next: {assessment['next_step']}",
            "",
            "## Traces",
            "",
            "- `results/symmetric_reward_poc_control_trace.csv`",
            "- `results/symmetric_reward_poc_variant_trace.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--retrain-control", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()

    protocol = load_protocol()
    cfg = protocol["symmetric_reward_poc"]
    n_eval = args.eval_episodes or cfg.get("eval_episodes", 100)
    seed_base = cfg.get("eval_seed_base", 95000)

    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
    control_path = ckpt_dir / "masked_ppo_multifault_control_star.zip"
    variant_path = ckpt_dir / "masked_ppo_multifault_symmetric_star.zip"
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"

    if args.skip_train and control_path.with_suffix(".zip").exists() and not args.retrain_control:
        print("Loading control model...")
        control_model = MaskablePPO.load(str(control_path.with_suffix("")))
    else:
        print("Training control (original reward)...")
        control_model = train_model(protocol, reward_mode="original", out_path=control_path)

    if args.skip_train and variant_path.with_suffix(".zip").exists():
        print("Loading variant model...")
        variant_model = MaskablePPO.load(str(variant_path.with_suffix("")))
    else:
        print("Training variant R (symmetric reward)...")
        variant_model = train_model(protocol, reward_mode="symmetric", out_path=variant_path)

    print("Evaluating control...")
    control = evaluate_model(
        control_model,
        protocol,
        n_episodes=n_eval,
        seed_base=seed_base,
        agent_label="masked_ppo_control",
    )
    print("Evaluating variant R...")
    variant = evaluate_model(
        variant_model,
        protocol,
        n_episodes=n_eval,
        seed_base=seed_base,
        agent_label="masked_ppo_symmetric",
    )

    write_trace_csv(results / "symmetric_reward_poc_control_trace.csv", control.pop("records"))
    write_trace_csv(results / "symmetric_reward_poc_variant_trace.csv", variant.pop("records"))

    assessment = assess_poc(control, variant)
    report = build_report(control, variant, assessment, cfg)
    (reports / "symmetric_reward_poc.md").write_text(report, encoding="utf-8")

    summary = {
        "control": {k: v for k, v in control.items()},
        "variant": {k: v for k, v in variant.items()},
        "assessment": assessment,
    }
    (results / "symmetric_reward_poc.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Control DSR={control['decomposition']['exact_diagnosis_accuracy']:.0%} "
          f"Variant DSR={variant['decomposition']['exact_diagnosis_accuracy']:.0%}")
    print(f"Verdict: {assessment['verdict']}")
    print(f"Wrote {reports / 'symmetric_reward_poc.md'}")


if __name__ == "__main__":
    main()
