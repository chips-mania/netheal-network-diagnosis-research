"""Step 9.13-B — Diagnosis Gate ablation vs masked PPO control."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_mask import action_masks_for_env
from .action_trace import EpisodeSummary, StepRecord, aggregate_behavior, run_episode_with_policy, write_trace_csv
from .diagnosis_decomposition import aggregate_decomposition
from .env_utils import make_masked_base_env, make_masked_eval_env, metrics_env
from .metrics_utils import summarize_evaluator
from .protocol import PROJECT_ROOT, load_protocol
from .run_symmetric_reward_poc import _make_ppo_selector, set_seeds


def shortcut_rate(summaries: List[EpisodeSummary], records: List[StepRecord]) -> Dict[str, float]:
    """scan → diagnose in <=2 steps, with no diagnostic tool in between."""
    by_ep: Dict[int, List[StepRecord]] = defaultdict(list)
    for rec in records:
        by_ep[rec.episode_id].append(rec)

    n = max(1, len(summaries))
    two_step = 0
    scan_then_dx = 0
    for s in summaries:
        recs = sorted(by_ep.get(s.episode_id, []), key=lambda r: r.step)
        if s.steps <= 2:
            two_step += 1
        types = [r.action_type for r in recs]
        cats = [r.action_category for r in recs]
        if len(recs) <= 2 and "scan_network" in types and "diagnosis" in cats:
            if not any(c == "diagnostic" for c in cats):
                scan_then_dx += 1

    return {
        "two_step_or_less": two_step / n,
        "scan_then_diagnose": scan_then_dx / n,
        "shortcut_rate": scan_then_dx / n,
    }


def train_gate_model(protocol: dict, out_path: Path) -> MaskablePPO:
    cfg = protocol["diagnosis_gate_poc"]
    set_seeds(cfg["policy_seed"])
    env = make_masked_base_env(
        cfg["topology_types"],
        protocol=protocol,
        diagnosis_gate=True,
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


def evaluate_model(
    model: MaskablePPO,
    protocol: dict,
    *,
    diagnosis_gate: bool,
    n_episodes: int,
    seed_base: int,
    agent_label: str,
) -> Dict:
    cfg = protocol["diagnosis_gate_poc"]
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
            diagnosis_gate=diagnosis_gate,
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
    sc = shortcut_rate(summaries, records)

    return {
        "stats": stats,
        "behavior": behavior,
        "decomposition": {
            "fault_type_accuracy": decomp["fault_type_accuracy"],
            "location_accuracy": decomp["location_accuracy"],
            "exact_diagnosis_accuracy": decomp["exact_diagnosis_accuracy"],
        },
        "per_fault_dsr": behavior.get("fault_type_dsr", {}),
        "shortcut": sc,
        "records": records,
    }


def assess_gate(control: Dict, variant: Dict) -> Dict:
    c_dsr = control["decomposition"]["exact_diagnosis_accuracy"]
    v_dsr = variant["decomposition"]["exact_diagnosis_accuracy"]
    delta = v_dsr - c_dsr
    dummy_tool = (
        variant["behavior"]["avg_tool_actions_before_diagnosis"] >= 2.0
        and variant["shortcut"]["shortcut_rate"] < 0.10
        and variant["stats"]["avg_steps_all"] <= 3.2
        and v_dsr <= c_dsr + 0.05
    )
    improved_nonlink = sum(
        1
        for ft in ("device_failure", "performance_degradation", "misconfiguration")
        if variant["per_fault_dsr"].get(ft, 0) > control["per_fault_dsr"].get(ft, 0) + 0.05
    )
    success = (
        variant["shortcut"]["shortcut_rate"] < 0.10
        and variant["behavior"]["avg_tool_actions_before_diagnosis"] >= 2.0
        and (delta >= 0.10 or improved_nonlink >= 1)
        and not dummy_tool
    )
    if dummy_tool:
        next_step = "Gate only forced a dummy diagnostic; proceed to 9.13-C Factorized Diagnosis."
        verdict = "dummy_tool_collapse"
    elif success:
        next_step = "Gate reduced shortcut with DSR gain — consider 100k gate variant, then Mesh probe."
        verdict = "promising"
    else:
        next_step = "Gate inconclusive; proceed to 9.13-C Factorized Diagnosis."
        verdict = "inconclusive_or_fail"

    return {
        "dsr_delta": delta,
        "shortcut_control": control["shortcut"]["shortcut_rate"],
        "shortcut_gate": variant["shortcut"]["shortcut_rate"],
        "nonlink_improved": improved_nonlink,
        "dummy_tool_collapse": dummy_tool,
        "verdict": verdict,
        "next_step": next_step,
    }


def build_report(control: Dict, variant: Dict, assessment: Dict, cfg: dict) -> str:
    c, v = control, variant
    return "\n".join(
        [
            "# Step 9.13-B — Diagnosis Gate Ablation",
            "",
            "> **Protocol variant G** — diagnosis blocked until ≥1 diagnostic action.",
            "> Ablation only; not proposed as the final topology-generalization protocol.",
            "",
            f"Train: Star, 6 devices, 4 faults, MaskablePPO, {cfg['timesteps']} steps, seed {cfg['policy_seed']}",
            f"Eval: {cfg.get('eval_episodes', 100)} episodes, seed base {cfg.get('eval_seed_base', 96000)}",
            "",
            "## Comparison",
            "",
            "| Metric | Control (Masked) | Variant G (Gate) |",
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
            f"| **Shortcut Rate** | {c['shortcut']['shortcut_rate']:.0%} | "
            f"{v['shortcut']['shortcut_rate']:.0%} |",
            f"| Coverage | {c['stats']['topology_coverage']:.2f} | {v['stats']['topology_coverage']:.2f} |",
            "",
            "### Per-fault DSR",
            "",
            "| Fault | Control | Gate |",
            "|---|---:|---:|",
            *[
                f"| {ft} | {c['per_fault_dsr'].get(ft, 0):.0%} | {v['per_fault_dsr'].get(ft, 0):.0%} |"
                for ft in (
                    "link_failure",
                    "device_failure",
                    "performance_degradation",
                    "misconfiguration",
                )
            ],
            "",
            "## Assessment",
            "",
            f"- DSR delta: **{assessment['dsr_delta']:+.0%}**",
            f"- Shortcut: {assessment['shortcut_control']:.0%} → {assessment['shortcut_gate']:.0%}",
            f"- Non-link faults improved (≥5%p): **{assessment['nonlink_improved']}/3**",
            f"- Dummy-tool collapse: **{assessment['dummy_tool_collapse']}**",
            f"- Verdict: **{assessment['verdict']}**",
            f"- Next: {assessment['next_step']}",
            "",
            "## Traces",
            "",
            "- `results/diagnosis_gate_poc_control_trace.csv`",
            "- `results/diagnosis_gate_poc_variant_trace.csv`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()

    protocol = load_protocol()
    cfg = protocol["diagnosis_gate_poc"]
    n_eval = args.eval_episodes or cfg.get("eval_episodes", 100)
    seed_base = cfg.get("eval_seed_base", 96000)

    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
    control_path = ckpt_dir / "masked_ppo_multifault_control_star.zip"
    gate_path = ckpt_dir / "masked_ppo_multifault_gate_star.zip"
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"

    if args.skip_train and control_path.with_suffix(".zip").exists():
        print("Loading control (masked, no gate)...")
        control_model = MaskablePPO.load(str(control_path.with_suffix("")))
    else:
        print("Training control (masked, no gate)...")
        set_seeds(cfg["policy_seed"])
        env_c = make_masked_base_env(cfg["topology_types"], protocol=protocol, diagnosis_gate=False)
        check_env(env_c, warn=True)
        control_model = MaskablePPO("MultiInputPolicy", env_c, verbose=1, seed=cfg["policy_seed"])
        control_model.learn(total_timesteps=cfg["timesteps"])
        control_path.parent.mkdir(parents=True, exist_ok=True)
        control_model.save(str(control_path.with_suffix("")))
        env_c.close()

    if args.skip_train and gate_path.with_suffix(".zip").exists():
        print("Loading gate model...")
        gate_model = MaskablePPO.load(str(gate_path.with_suffix("")))
    else:
        print("Training variant G (diagnosis gate)...")
        gate_model = train_gate_model(protocol, gate_path)

    print("Evaluating control...")
    control = evaluate_model(
        control_model,
        protocol,
        diagnosis_gate=False,
        n_episodes=n_eval,
        seed_base=seed_base,
        agent_label="masked_ppo_control",
    )
    print("Evaluating gate...")
    variant = evaluate_model(
        gate_model,
        protocol,
        diagnosis_gate=True,
        n_episodes=n_eval,
        seed_base=seed_base,
        agent_label="masked_ppo_gate",
    )

    write_trace_csv(results / "diagnosis_gate_poc_control_trace.csv", control.pop("records"))
    write_trace_csv(results / "diagnosis_gate_poc_variant_trace.csv", variant.pop("records"))

    assessment = assess_gate(control, variant)
    report = build_report(control, variant, assessment, cfg)
    (reports / "diagnosis_gate_poc.md").write_text(report, encoding="utf-8")
    (results / "diagnosis_gate_poc.json").write_text(
        json.dumps(
            {"control": control, "variant": variant, "assessment": assessment},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"Control DSR={control['decomposition']['exact_diagnosis_accuracy']:.0%} "
        f"Gate DSR={variant['decomposition']['exact_diagnosis_accuracy']:.0%} "
        f"Shortcut {assessment['shortcut_control']:.0%}→{assessment['shortcut_gate']:.0%}"
    )
    print(f"Verdict: {assessment['verdict']}")
    print(f"Wrote {reports / 'diagnosis_gate_poc.md'}")


if __name__ == "__main__":
    main()
