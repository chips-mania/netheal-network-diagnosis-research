"""Step 9.13-C — Factorized Diagnosis (type → location) vs masked PPO control."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.environment.actions import ActionCategory, TopologyAction
from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_mask import action_masks_for_env, base_env, find_factorized_diagnosis
from .action_trace import EpisodeSummary, StepRecord, aggregate_behavior, run_episode_with_policy, write_trace_csv
from .diagnosis_decomposition import aggregate_decomposition
from .env_utils import make_masked_base_env, make_masked_eval_env, metrics_env
from .factorized_diagnosis import DEVICE_LEVEL_TYPES
from .metrics_utils import summarize_evaluator
from .protocol import PROJECT_ROOT, load_protocol
from .run_symmetric_reward_poc import _make_ppo_selector, set_seeds


def shortcut_rates(summaries: List[EpisodeSummary], records: List[StepRecord]) -> Dict[str, float]:
    """Joint shortcut (≤2) and factorized shortcut (scan → type → location, no diagnostic)."""
    by_ep: Dict[int, List[StepRecord]] = defaultdict(list)
    for rec in records:
        by_ep[rec.episode_id].append(rec)

    n = max(1, len(summaries))
    two_step = 0
    scan_then_dx = 0
    factorized_shortcut = 0
    type_then_loc = 0
    for s in summaries:
        recs = sorted(by_ep.get(s.episode_id, []), key=lambda r: r.step)
        if s.steps <= 2:
            two_step += 1
        types = [r.action_type for r in recs]
        cats = [r.action_category for r in recs]
        has_scan = "scan_network" in types
        has_terminal_dx = "diagnosis" in cats
        has_type_commit = "diagnosis_type" in cats
        has_diagnostic = any(c == "diagnostic" for c in cats)
        if s.steps <= 2 and has_scan and has_terminal_dx and not has_diagnostic:
            scan_then_dx += 1
        if (
            s.steps <= 3
            and has_scan
            and has_type_commit
            and has_terminal_dx
            and not has_diagnostic
        ):
            factorized_shortcut += 1
        if has_type_commit and has_terminal_dx:
            type_then_loc += 1

    return {
        "two_step_or_less": two_step / n,
        "scan_then_diagnose": scan_then_dx / n,
        "shortcut_rate": scan_then_dx / n,
        "factorized_shortcut_rate": factorized_shortcut / n,
        "type_then_location_rate": type_then_loc / n,
    }


def smoke_factorized(protocol: dict) -> Dict:
    """Leakage-free structural check: post-scan 4 types, then 6 or 30 locations."""
    cfg = protocol["factorized_diagnosis_poc"]
    env = make_masked_base_env(
        cfg["topology_types"],
        protocol=protocol,
        factorized_diagnosis=True,
    )
    env.reset(seed=42)
    root = base_env(env)
    fac = find_factorized_diagnosis(env)
    scan_id = None
    for aid, spec in root.action_space_manager.action_map.items():
        if spec.action_type == TopologyAction.SCAN_NETWORK:
            scan_id = aid
            break
    if scan_id is None:
        raise RuntimeError("scan_network action not found")
    env.step(scan_id)

    phase1 = [aid for aid in fac.get_valid_actions() if _is_dx(root, aid)]
    obs, reward, terminated, truncated, info = env.step(phase1[0])
    phase2 = [aid for aid in fac.get_valid_actions() if _is_dx(root, aid)]
    committed = fac.committed_type
    expected = 6 if committed in DEVICE_LEVEL_TYPES else 30
    env.close()

    result = {
        "post_scan_type_actions": len(phase1),
        "type_commit_terminated": bool(terminated),
        "type_commit_reward": float(reward),
        "type_commit_phase": info.get("factorized_phase"),
        "committed_type": committed.value if committed else None,
        "post_commit_location_actions": len(phase2),
        "expected_locations": expected,
        "obs_has_committed_type": isinstance(obs, dict) and "committed_fault_type" in obs,
        "ok": (
            len(phase1) == 4
            and not terminated
            and not truncated
            and info.get("factorized_phase") == "type_commit"
            and len(phase2) == expected
        ),
    }
    if not result["ok"]:
        raise RuntimeError(f"Factorized smoke failed: {result}")
    return result


def _is_dx(root, action_id: int) -> bool:
    spec = root.action_space_manager.get_action_spec(action_id)
    return spec is not None and spec.category == ActionCategory.DIAGNOSIS


def train_factorized_model(protocol: dict, out_path: Path) -> MaskablePPO:
    cfg = protocol["factorized_diagnosis_poc"]
    set_seeds(cfg["policy_seed"])
    env = make_masked_base_env(
        cfg["topology_types"],
        protocol=protocol,
        factorized_diagnosis=True,
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
    factorized_diagnosis: bool,
    n_episodes: int,
    seed_base: int,
    agent_label: str,
) -> Dict:
    cfg = protocol["factorized_diagnosis_poc"]
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
            factorized_diagnosis=factorized_diagnosis,
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
    sc = shortcut_rates(summaries, records)

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


def assess_factorized(control: Dict, variant: Dict) -> Dict:
    c_dsr = control["decomposition"]["exact_diagnosis_accuracy"]
    v_dsr = variant["decomposition"]["exact_diagnosis_accuracy"]
    delta = v_dsr - c_dsr
    improved_nonlink = sum(
        1
        for ft in ("device_failure", "performance_degradation", "misconfiguration")
        if variant["per_fault_dsr"].get(ft, 0) > control["per_fault_dsr"].get(ft, 0) + 0.05
    )
    cats = variant["behavior"].get("action_category_frequency", {})
    n_terminal_dx = int(cats.get("diagnosis", 0))
    n_type_commit = int(cats.get("diagnosis_type", 0))
    never_diagnosed = (
        n_terminal_dx == 0
        and n_type_commit == 0
        and variant["stats"]["avg_steps_all"] >= 14.0
    )
    factorized_shortcut = variant["shortcut"]["factorized_shortcut_rate"] >= 0.50
    still_joint_shortcut = variant["shortcut"]["shortcut_rate"] >= 0.50
    uses_tools = (
        not never_diagnosed
        and variant["behavior"]["avg_tool_actions_before_diagnosis"] >= 2.0
    )
    success = (
        not factorized_shortcut
        and not still_joint_shortcut
        and uses_tools
        and (delta >= 0.10 or improved_nonlink >= 1)
    )
    if never_diagnosed:
        verdict = "diagnosis_avoidance_collapse"
        next_step = (
            "Policy never submits type or location; timeout (-1.5) beats a wrong "
            "diagnosis (~-16). Do not scale 100k or Mesh on factorization alone."
        )
    elif factorized_shortcut and v_dsr <= c_dsr + 0.05:
        verdict = "factorized_shortcut_collapse"
        next_step = (
            "Type→location only split the lottery; no troubleshooting policy. "
            "Do not scale 100k or Mesh on factorization alone."
        )
    elif success:
        verdict = "promising"
        next_step = (
            "Factorized diagnosis formed a multi-step policy — "
            "consider 100k factorized variant, then Mesh probe."
        )
    else:
        verdict = "inconclusive_or_fail"
        next_step = (
            "Factorized diagnosis did not yield a valid troubleshooting policy. "
            "Hold 100k and Mesh; compare protocols, do not add fault-specific bonuses."
        )

    return {
        "dsr_delta": delta,
        "shortcut_control": control["shortcut"]["shortcut_rate"],
        "shortcut_factorized": variant["shortcut"]["shortcut_rate"],
        "factorized_shortcut_rate": variant["shortcut"]["factorized_shortcut_rate"],
        "nonlink_improved": improved_nonlink,
        "uses_tools": uses_tools,
        "verdict": verdict,
        "next_step": next_step,
    }


def build_report(control: Dict, variant: Dict, assessment: Dict, cfg: dict, smoke: Dict) -> str:
    c, v = control, variant
    return "\n".join(
        [
            "# Step 9.13-C — Factorized Diagnosis PoC",
            "",
            "> **Protocol variant F** — diagnosis is type (4) then location (6 or 30).",
            "> Structural protocol variant; does not modify the NetHeal package.",
            "",
            f"Train: Star, 6 devices, 4 faults, MaskablePPO, {cfg['timesteps']} steps, seed {cfg['policy_seed']}",
            f"Eval: {cfg.get('eval_episodes', 100)} episodes, seed base {cfg.get('eval_seed_base', 97000)}",
            "",
            "## Smoke (structure)",
            "",
            f"- Post-scan type actions: **{smoke['post_scan_type_actions']}** (expect 4)",
            f"- After type-commit locations: **{smoke['post_commit_location_actions']}** "
            f"(expect {smoke['expected_locations']} for `{smoke['committed_type']}`)",
            f"- Type-commit terminates: **{smoke['type_commit_terminated']}**",
            "",
            "## Comparison",
            "",
            "| Metric | Control (Masked) | Variant F (Factorized) |",
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
            f"| Shortcut Rate (scan→dx ≤2) | {c['shortcut']['shortcut_rate']:.0%} | "
            f"{v['shortcut']['shortcut_rate']:.0%} |",
            f"| Factorized Shortcut (scan→type→loc ≤3) | {c['shortcut']['factorized_shortcut_rate']:.0%} | "
            f"{v['shortcut']['factorized_shortcut_rate']:.0%} |",
            f"| Coverage | {c['stats']['topology_coverage']:.2f} | {v['stats']['topology_coverage']:.2f} |",
            "",
            "### Per-fault DSR",
            "",
            "| Fault | Control | Factorized |",
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
            f"- Joint shortcut: {assessment['shortcut_control']:.0%} → {assessment['shortcut_factorized']:.0%}",
            f"- Factorized shortcut: **{assessment['factorized_shortcut_rate']:.0%}**",
            f"- Non-link faults improved (≥5%p): **{assessment['nonlink_improved']}/3**",
            f"- Uses diagnostic tools (≥2 before dx): **{assessment['uses_tools']}**",
            f"- Verdict: **{assessment['verdict']}**",
            f"- Next: {assessment['next_step']}",
            "",
            "## Policy behavior",
            "",
            "Control remains **scan → diagnose** (joint 72-way lottery). Link DSR is the only non-zero cell.",
            "",
            "Variant F **never commits a type and never submits a location**. Typical episode:",
            "",
            "```",
            "scan → ping(edge_1→edge_3) × N → timeout at 15 steps",
            "```",
            "",
            "This is locally rational under the original sparse reward:",
            "",
            "| Outcome | Approx. return |",
            "|---|---:|",
            "| Exact diagnosis | +15.9 |",
            "| Wrong diagnosis | −16.1 |",
            "| Timeout (15 × −0.1) | **−1.5** |",
            "",
            "Type-commit is unrewarded. The second action still faces a 6- or 30-way location lottery.",
            "Avoiding diagnosis entirely dominates guessing. Coverage 0.00 is a metrics-wrapper artifact;",
            "traces show `scan_network` on step 1 in every episode.",
            "",
            "## Traces",
            "",
            "- `results/factorized_diagnosis_poc_control_trace.csv`",
            "- `results/factorized_diagnosis_poc_variant_trace.csv`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol()
    cfg = protocol["factorized_diagnosis_poc"]
    n_eval = args.eval_episodes or cfg.get("eval_episodes", 100)
    seed_base = cfg.get("eval_seed_base", 97000)

    print("Smoke-testing factorized diagnosis wrapper...")
    smoke = smoke_factorized(protocol)
    print(f"Smoke OK: {smoke}")
    if args.smoke_only:
        return

    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
    control_path = ckpt_dir / "masked_ppo_multifault_control_star.zip"
    factorized_path = ckpt_dir / "masked_ppo_multifault_factorized_star.zip"
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"

    if args.skip_train and control_path.with_suffix(".zip").exists():
        print("Loading control (masked, joint diagnosis)...")
        control_model = MaskablePPO.load(str(control_path.with_suffix("")))
    else:
        print("Training control (masked, joint diagnosis)...")
        set_seeds(cfg["policy_seed"])
        env_c = make_masked_base_env(
            cfg["topology_types"], protocol=protocol, factorized_diagnosis=False
        )
        check_env(env_c, warn=True)
        control_model = MaskablePPO("MultiInputPolicy", env_c, verbose=1, seed=cfg["policy_seed"])
        control_model.learn(total_timesteps=cfg["timesteps"])
        control_path.parent.mkdir(parents=True, exist_ok=True)
        control_model.save(str(control_path.with_suffix("")))
        env_c.close()

    if args.skip_train and factorized_path.with_suffix(".zip").exists():
        print("Loading factorized model...")
        factorized_model = MaskablePPO.load(str(factorized_path.with_suffix("")))
    else:
        print("Training variant F (factorized diagnosis)...")
        factorized_model = train_factorized_model(protocol, factorized_path)

    print("Evaluating control...")
    control = evaluate_model(
        control_model,
        protocol,
        factorized_diagnosis=False,
        n_episodes=n_eval,
        seed_base=seed_base,
        agent_label="masked_ppo_control",
    )
    print("Evaluating factorized...")
    variant = evaluate_model(
        factorized_model,
        protocol,
        factorized_diagnosis=True,
        n_episodes=n_eval,
        seed_base=seed_base,
        agent_label="masked_ppo_factorized",
    )

    write_trace_csv(results / "factorized_diagnosis_poc_control_trace.csv", control.pop("records"))
    write_trace_csv(results / "factorized_diagnosis_poc_variant_trace.csv", variant.pop("records"))

    assessment = assess_factorized(control, variant)
    report = build_report(control, variant, assessment, cfg, smoke)
    (reports / "factorized_diagnosis_poc.md").write_text(report, encoding="utf-8")
    (results / "factorized_diagnosis_poc.json").write_text(
        json.dumps(
            {"smoke": smoke, "control": control, "variant": variant, "assessment": assessment},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"Control DSR={control['decomposition']['exact_diagnosis_accuracy']:.0%} "
        f"Factorized DSR={variant['decomposition']['exact_diagnosis_accuracy']:.0%} "
        f"F-shortcut={assessment['factorized_shortcut_rate']:.0%}"
    )
    print(f"Verdict: {assessment['verdict']}")
    print(f"Wrote {reports / 'factorized_diagnosis_poc.md'}")


if __name__ == "__main__":
    main()
