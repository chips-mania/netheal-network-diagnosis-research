"""Step E1 — Pure PBRS vs masked PPO control (seed 1 Go/No-Go)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_mask import action_masks_for_env
from .action_trace import aggregate_behavior, run_episode_with_policy, write_trace_csv
from .diagnosis_decomposition import aggregate_decomposition, location_match
from .env_utils import make_masked_base_env, make_masked_eval_env, metrics_env
from .metrics_utils import summarize_evaluator
from .pbrs import ALPHA, GAMMA, find_pbrs
from .pbrs_unit_tests import run_pbrs_unit_tests
from .protocol import PROJECT_ROOT, load_protocol
from .run_diagnosis_gate_poc import shortcut_rate
from .run_symmetric_reward_poc import _make_ppo_selector, set_seeds

NONLINK = ("device_failure", "performance_degradation", "misconfiguration")


def evidence_from_episodes(rows: List[Dict]) -> Dict[str, float]:
    nonlink = [r for r in rows if r["fault_type"] != "link_failure"]
    n = max(1, len(nonlink))
    acquired = [r for r in nonlink if r["phi_before_dx"] >= 1]
    consistent = 0
    for r in acquired:
        ok = True
        if r["type_bit"] and r["predicted_type"] != r["fault_type"]:
            ok = False
        if r["loc_bit"] and not location_match(
            r["fault_location"], r["predicted_location"], r["fault_type"]
        ):
            ok = False
        if ok:
            consistent += 1
    early = sum(1 for r in nonlink if r.get("first_phi_step") is not None and r["first_phi_step"] <= 7)
    late = sum(1 for r in nonlink if r.get("first_phi_step") is not None and r["first_phi_step"] >= 8)
    return {
        "nonlink_episodes": len(nonlink),
        "evidence_acquisition": len(acquired) / n,
        "evidence_consistency": consistent / max(1, len(acquired)),
        "nonlink_dsr": sum(1 for r in nonlink if r["success"]) / n,
        "early_acquisition_count": early,
        "late_acquisition_count": late,
    }


def evaluate_model(
    model: MaskablePPO,
    protocol: dict,
    *,
    n_episodes: int,
    seed_base: int,
    agent_label: str,
) -> Dict:
    cfg = protocol["e1_pbrs_poc"]
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    evaluator = CompetitionEvaluator()
    records: List = []
    summaries: List = []
    rows: List[Dict] = []

    for i in range(n_episodes):
        fault = faults[i % len(faults)]
        seed = seed_base + i
        env = make_masked_eval_env(
            cfg["topology_types"],
            fault_types=[fault],
            protocol=protocol,
            pbrs=True,
            pbrs_apply_shaping=False,
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
        tracker = find_pbrs(env)
        bits = tracker.bits_before_diagnosis or tracker.bits.snapshot()
        phi_dx = tracker.phi_before_diagnosis
        if phi_dx is None:
            phi_dx = tracker.bits.phi
        first_phi = None
        for rec in ep_records:
            if rec.action_category == "diagnostic":
                first_phi = rec.step
                break
        rows.append(
            {
                "fault_type": summary.fault_type,
                "fault_location": summary.fault_location,
                "success": summary.success,
                "predicted_type": summary.predicted_type,
                "predicted_location": summary.predicted_location,
                "phi_before_dx": int(phi_dx),
                "type_bit": int(bits.get("type_bit", 0)),
                "loc_bit": int(bits.get("loc_bit", 0)),
                "first_phi_step": first_phi,
            }
        )
        records.extend(ep_records)
        summaries.append(summary)
        env.close()

    stats = summarize_evaluator(evaluator)
    behavior = aggregate_behavior(summaries, records)
    decomp = aggregate_decomposition(summaries)
    ev = evidence_from_episodes(rows)
    return {
        "stats": stats,
        "behavior": behavior,
        "decomposition": {
            "fault_type_accuracy": decomp["fault_type_accuracy"],
            "location_accuracy": decomp["location_accuracy"],
            "exact_diagnosis_accuracy": decomp["exact_diagnosis_accuracy"],
        },
        "per_fault_dsr": behavior.get("fault_type_dsr", {}),
        "shortcut": shortcut_rate(summaries, records),
        "evidence": ev,
        "records": records,
    }


def go_nogo(control: Dict, variant: Dict) -> Dict:
    d_nl = variant["evidence"]["nonlink_dsr"] - control["evidence"]["nonlink_dsr"]
    d_acq = variant["evidence"]["evidence_acquisition"] - control["evidence"]["evidence_acquisition"]
    nogo = d_nl < 0.03 and d_acq < 0.05
    return {
        "nonlink_dsr_delta": d_nl,
        "acquisition_delta": d_acq,
        "nogo": nogo,
        "decision": "no_go" if nogo else "go",
        "rule": "No-Go if non-link DSR Δ < +3%p AND Evidence Acquisition Δ < +5%p",
    }


def assess(control: Dict, variant: Dict, gate: Dict) -> Dict:
    c_nl = control["evidence"]["nonlink_dsr"]
    v_nl = variant["evidence"]["nonlink_dsr"]
    c_cons = control["evidence"]["evidence_consistency"]
    v_cons = variant["evidence"]["evidence_consistency"]
    link_only = (
        variant["per_fault_dsr"].get("link_failure", 0) > control["per_fault_dsr"].get("link_failure", 0)
        and all(
            variant["per_fault_dsr"].get(ft, 0) <= control["per_fault_dsr"].get(ft, 0) + 0.02
            for ft in NONLINK
        )
    )
    success = (v_nl > c_nl) and (v_cons > c_cons) and (not link_only)
    if gate["nogo"]:
        verdict = "no_go_seed1"
        next_step = "Dense credit alone did not move non-link / acquisition. Hold seed 2/3. Consider E2."
    elif success:
        verdict = "promising_seed1"
        next_step = "Seed 1 Go on the numeric rule and 7.1 pattern. Run seed 2/3 before claiming success."
    else:
        verdict = "go_but_not_success"
        next_step = "Seed 1 moved enough to open 3-seed, but 7.1 success criteria not met yet."
    return {
        "nonlink_dsr_delta": v_nl - c_nl,
        "consistency_delta": v_cons - c_cons,
        "link_dsr_delta": variant["per_fault_dsr"].get("link_failure", 0)
        - control["per_fault_dsr"].get("link_failure", 0),
        "link_only_pattern": link_only,
        "verdict": verdict,
        "next_step": next_step,
    }


def build_report(control: Dict, variant: Dict, gate: Dict, assessment: Dict, cfg: dict, tests: dict) -> str:
    c, v = control, variant
    return "\n".join(
        [
            "# E1 — Pure Potential-Based Evidence Shaping (seed 1)",
            "",
            "> Simulator ground truth was used only to compute the training-time shaping potential; "
            "it was not exposed to the policy observation.",
            "",
            f"Train: Star, 6 devices, 4 faults, MaskablePPO, {cfg['timesteps']} steps, seed {cfg['policy_seed']}",
            f"Eval: {cfg.get('eval_episodes', 100)} episodes, seed base {cfg.get('eval_seed_base', 98000)}",
            f"α={ALPHA}, γ={GAMMA}. Eval shaping off. Evidence metrics = non-link only.",
            "",
            f"Unit tests: {tests}",
            "",
            "## Primary (seed 1)",
            "",
            "| Metric | Control (Masked) | E1 PBRS |",
            "|---|---:|---:|",
            f"| Exact DSR | {c['decomposition']['exact_diagnosis_accuracy']:.0%} | "
            f"{v['decomposition']['exact_diagnosis_accuracy']:.0%} |",
            f"| **Non-link DSR** | {c['evidence']['nonlink_dsr']:.0%} | {v['evidence']['nonlink_dsr']:.0%} |",
            f"| Type Acc | {c['decomposition']['fault_type_accuracy']:.0%} | "
            f"{v['decomposition']['fault_type_accuracy']:.0%} |",
            f"| Loc Acc | {c['decomposition']['location_accuracy']:.0%} | "
            f"{v['decomposition']['location_accuracy']:.0%} |",
            f"| Evidence Acquisition (non-link) | {c['evidence']['evidence_acquisition']:.0%} | "
            f"{v['evidence']['evidence_acquisition']:.0%} |",
            f"| Evidence Consistency (non-link) | {c['evidence']['evidence_consistency']:.0%} | "
            f"{v['evidence']['evidence_consistency']:.0%} |",
            f"| Shortcut Rate | {c['shortcut']['shortcut_rate']:.0%} | {v['shortcut']['shortcut_rate']:.0%} |",
            f"| AvgSteps | {c['stats']['avg_steps_all']:.1f} | {v['stats']['avg_steps_all']:.1f} |",
            "",
            "| Variant | Link DSR | Link DSR Δ | Device | Perf | Misconfig |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Control | {c['per_fault_dsr'].get('link_failure', 0):.0%} | — | "
            f"{c['per_fault_dsr'].get('device_failure', 0):.0%} | "
            f"{c['per_fault_dsr'].get('performance_degradation', 0):.0%} | "
            f"{c['per_fault_dsr'].get('misconfiguration', 0):.0%} |",
            f"| E1 | {v['per_fault_dsr'].get('link_failure', 0):.0%} | "
            f"{assessment['link_dsr_delta']:+.0%} | "
            f"{v['per_fault_dsr'].get('device_failure', 0):.0%} | "
            f"{v['per_fault_dsr'].get('performance_degradation', 0):.0%} | "
            f"{v['per_fault_dsr'].get('misconfiguration', 0):.0%} |",
            "",
            "## Go/No-Go (7.2)",
            "",
            f"- Non-link DSR Δ: **{gate['nonlink_dsr_delta']:+.1%}** (No-Go if < +3%p)",
            f"- Evidence Acquisition Δ: **{gate['acquisition_delta']:+.1%}** (No-Go if < +5%p)",
            f"- Decision: **{gate['decision']}**",
            f"- {gate['rule']}",
            "",
            "## Assessment",
            "",
            f"- Verdict: **{assessment['verdict']}**",
            f"- Next: {assessment['next_step']}",
            "",
            "## Traces",
            "",
            "- `results/e1_pbrs_poc_control_trace.csv`",
            "- `results/e1_pbrs_poc_variant_trace.csv`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--unit-tests-only", action="store_true")
    args = parser.parse_args()

    print("E1 unit tests...")
    tests = run_pbrs_unit_tests()
    print(f"Unit tests passed: {tests}")
    if args.unit_tests_only:
        return

    protocol = load_protocol()
    cfg = protocol["e1_pbrs_poc"]
    n_eval = args.eval_episodes or cfg.get("eval_episodes", 100)
    seed_base = cfg.get("eval_seed_base", 98000)

    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
    control_path = ckpt_dir / "masked_ppo_multifault_control_star.zip"
    e1_path = ckpt_dir / "masked_ppo_multifault_e1_pbrs_star.zip"
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"

    if args.skip_train and control_path.with_suffix(".zip").exists():
        print("Loading control...")
        control_model = MaskablePPO.load(str(control_path.with_suffix("")))
    else:
        print("Training control (masked, no PBRS)...")
        set_seeds(cfg["policy_seed"])
        env_c = make_masked_base_env(cfg["topology_types"], protocol=protocol)
        check_env(env_c, warn=True)
        control_model = MaskablePPO("MultiInputPolicy", env_c, verbose=1, seed=cfg["policy_seed"])
        control_model.learn(total_timesteps=cfg["timesteps"])
        control_path.parent.mkdir(parents=True, exist_ok=True)
        control_model.save(str(control_path.with_suffix("")))
        env_c.close()

    if args.skip_train and e1_path.with_suffix(".zip").exists():
        print("Loading E1...")
        e1_model = MaskablePPO.load(str(e1_path.with_suffix("")))
    else:
        print("Training E1 (PBRS shaping on)...")
        set_seeds(cfg["policy_seed"])
        env_e = make_masked_base_env(
            cfg["topology_types"], protocol=protocol, pbrs=True, pbrs_apply_shaping=True
        )
        check_env(env_e, warn=True)
        e1_model = MaskablePPO("MultiInputPolicy", env_e, verbose=1, seed=cfg["policy_seed"])
        e1_model.learn(total_timesteps=cfg["timesteps"])
        e1_path.parent.mkdir(parents=True, exist_ok=True)
        e1_model.save(str(e1_path.with_suffix("")))
        env_e.close()

    print("Evaluating control...")
    control = evaluate_model(
        control_model, protocol, n_episodes=n_eval, seed_base=seed_base, agent_label="masked_ppo_control"
    )
    print("Evaluating E1...")
    variant = evaluate_model(
        e1_model, protocol, n_episodes=n_eval, seed_base=seed_base, agent_label="masked_ppo_e1_pbrs"
    )

    write_trace_csv(results / "e1_pbrs_poc_control_trace.csv", control.pop("records"))
    write_trace_csv(results / "e1_pbrs_poc_variant_trace.csv", variant.pop("records"))

    gate = go_nogo(control, variant)
    assessment = assess(control, variant, gate)
    report = build_report(control, variant, gate, assessment, cfg, tests)
    (reports / "e1_pbrs_poc.md").write_text(report, encoding="utf-8")
    (results / "e1_pbrs_poc.json").write_text(
        json.dumps(
            {
                "unit_tests": tests,
                "control": control,
                "variant": variant,
                "go_nogo": gate,
                "assessment": assessment,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        f"Control NL-DSR={control['evidence']['nonlink_dsr']:.0%} "
        f"E1 NL-DSR={variant['evidence']['nonlink_dsr']:.0%} "
        f"Go/No-Go={gate['decision']}"
    )
    print(f"Wrote {reports / 'e1_pbrs_poc.md'}")


if __name__ == "__main__":
    main()
