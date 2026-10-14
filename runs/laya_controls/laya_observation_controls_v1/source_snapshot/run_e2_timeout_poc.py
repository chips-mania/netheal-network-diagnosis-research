"""E2 — Timeout-as-incorrect-diagnosis vs masked control (seed 1 Go/No-Go)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.environment.actions import ActionCategory, DiagnosticAction, TopologyAction
from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_mask import base_env
from .action_trace import aggregate_behavior, run_episode_with_policy, write_trace_csv
from .controlled_faults import inject_controlled_fault
from .diagnosis_decomposition import aggregate_decomposition
from .env_utils import make_base_env, make_masked_base_env, make_masked_eval_env, metrics_env
from .metrics_utils import summarize_evaluator
from .protocol import PROJECT_ROOT, load_protocol
from .run_diagnosis_gate_poc import shortcut_rate
from .run_symmetric_reward_poc import _make_ppo_selector, set_seeds
from .timeout_penalty import incorrect_diagnosis_penalty, wrap_with_timeout_penalty

NONLINK = ("device_failure", "performance_degradation", "misconfiguration")


def _find(root, pred) -> int:
    for aid, spec in root.action_space_manager.action_map.items():
        if pred(spec):
            return aid
    raise RuntimeError("action not found")


def run_e2_unit_tests() -> dict:
    protocol = load_protocol()
    env = wrap_with_timeout_penalty(
        make_base_env(["star"], fault_types=[FaultType.DEVICE_FAILURE], protocol=protocol)
    )
    env.reset(seed=42)
    root = base_env(env)
    inject_controlled_fault(root, FaultType.DEVICE_FAILURE)
    expected = incorrect_diagnosis_penalty(root)
    scan = _find(root, lambda s: s.action_type == TopologyAction.SCAN_NETWORK)
    ping = _find(
        root,
        lambda s: s.action_type == DiagnosticAction.PING
        and s.parameters.get("source") == "edge_1"
        and s.parameters.get("destination") == "edge_2",
    )
    dx = _find(
        root,
        lambda s: s.category == ActionCategory.DIAGNOSIS
        and s.action_type == FaultType.DEVICE_FAILURE
        and s.parameters.get("location") == "edge_0",
    )

    env.step(scan)
    _, _, term, trunc, info = env.step(dx)
    assert not info.get("timeout_penalty_applied"), info
    assert abs(info.get("timeout_penalty", 0.0)) < 1e-9
    assert term
    env.close()

    env = wrap_with_timeout_penalty(
        make_base_env(["star"], fault_types=[FaultType.DEVICE_FAILURE], protocol=protocol)
    )
    env.reset(seed=42)
    root = base_env(env)
    inject_controlled_fault(root, FaultType.DEVICE_FAILURE)
    expected = incorrect_diagnosis_penalty(root)
    scan = _find(root, lambda s: s.action_type == TopologyAction.SCAN_NETWORK)
    ping = _find(
        root,
        lambda s: s.action_type == DiagnosticAction.PING
        and s.parameters.get("source") == "edge_1"
        and s.parameters.get("destination") == "edge_2",
    )
    env.step(scan)
    last_info = {}
    for _ in range(14):
        _, _, term, trunc, last_info = env.step(ping)
    assert trunc and not term
    assert last_info.get("timeout_penalty_applied")
    if abs(last_info["timeout_penalty"] - expected) > 1e-6:
        raise AssertionError(f"timeout extra {last_info['timeout_penalty']} != {expected}")
    env.close()
    return {"diagnose_no_extra": "ok", "timeout_gets_incorrect_penalty": "ok", "expected": expected}


def evaluate_model(model: MaskablePPO, protocol: dict, *, n_episodes: int, seed_base: int, agent: str) -> Dict:
    cfg = protocol["e2_timeout_poc"]
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    evaluator = CompetitionEvaluator()
    records: List = []
    summaries: List = []
    for i in range(n_episodes):
        fault = faults[i % len(faults)]
        env = make_masked_eval_env(
            cfg["topology_types"], fault_types=[fault], protocol=protocol
        )
        recs, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed_base + i,
            agent=agent,
            choose_action=_make_ppo_selector(model),
        )
        m = metrics_env(env).last_episode_metrics
        if m:
            evaluator.add_episode_metrics(m)
        records.extend(recs)
        summaries.append(summary)
        env.close()
    behavior = aggregate_behavior(summaries, records)
    decomp = aggregate_decomposition(summaries)
    per = behavior.get("fault_type_dsr", {})
    nl = [s.success for s in summaries if s.fault_type in NONLINK]
    return {
        "stats": summarize_evaluator(evaluator),
        "behavior": behavior,
        "decomposition": {
            "fault_type_accuracy": decomp["fault_type_accuracy"],
            "location_accuracy": decomp["location_accuracy"],
            "exact_diagnosis_accuracy": decomp["exact_diagnosis_accuracy"],
        },
        "per_fault_dsr": per,
        "nonlink_dsr": float(sum(nl) / max(1, len(nl))),
        "shortcut": shortcut_rate(summaries, records),
        "records": records,
    }


def go_nogo(control: Dict, variant: Dict) -> Dict:
    d_nl = variant["nonlink_dsr"] - control["nonlink_dsr"]
    d_sc = variant["shortcut"]["shortcut_rate"] - control["shortcut"]["shortcut_rate"]
    nogo = d_nl < 0.03 and d_sc > -0.10
    return {
        "nonlink_dsr_delta": d_nl,
        "shortcut_delta": d_sc,
        "nogo": nogo,
        "decision": "no_go" if nogo else "go",
        "rule": "No-Go if non-link DSR Δ < +3%p AND shortcut Δ > −10%p",
    }


def build_report(c: Dict, v: Dict, gate: Dict, tests: dict, cfg: dict) -> str:
    return "\n".join(
        [
            "# E2 — Timeout-as-Incorrect-Diagnosis (seed 1)",
            "",
            "> One change only: timeout without diagnosis gets the original incorrect-diagnosis penalty.",
            "> Exact/wrong/partial on diagnose actions unchanged. Not mixed with E1/Gate/Factorized/Symmetric.",
            "> Graded/symmetric partial was already tested in 9.13-A and is not repeated here.",
            "",
            f"Train: Star, 6 devices, 4 faults, MaskablePPO, {cfg['timesteps']}, seed {cfg['policy_seed']}",
            f"Eval: {cfg.get('eval_episodes', 100)} ep, seed base {cfg.get('eval_seed_base', 99000)}",
            f"Unit tests: {tests}",
            "",
            "| Metric | Control | E2 |",
            "|---|---:|---:|",
            f"| Exact DSR | {c['decomposition']['exact_diagnosis_accuracy']:.0%} | "
            f"{v['decomposition']['exact_diagnosis_accuracy']:.0%} |",
            f"| **Non-link DSR** | {c['nonlink_dsr']:.0%} | {v['nonlink_dsr']:.0%} |",
            f"| Type Acc | {c['decomposition']['fault_type_accuracy']:.0%} | "
            f"{v['decomposition']['fault_type_accuracy']:.0%} |",
            f"| Loc Acc | {c['decomposition']['location_accuracy']:.0%} | "
            f"{v['decomposition']['location_accuracy']:.0%} |",
            f"| Shortcut Rate | {c['shortcut']['shortcut_rate']:.0%} | {v['shortcut']['shortcut_rate']:.0%} |",
            f"| AvgSteps | {c['stats']['avg_steps_all']:.1f} | {v['stats']['avg_steps_all']:.1f} |",
            "",
            "| Fault | Control | E2 |",
            "|---|---:|---:|",
            *[
                f"| {ft} | {c['per_fault_dsr'].get(ft, 0):.0%} | {v['per_fault_dsr'].get(ft, 0):.0%} |"
                for ft in ("link_failure",) + NONLINK
            ],
            "",
            "## Go/No-Go",
            "",
            f"- Non-link DSR Δ: **{gate['nonlink_dsr_delta']:+.1%}**",
            f"- Shortcut Δ: **{gate['shortcut_delta']:+.1%}**",
            f"- Decision: **{gate['decision']}**",
            f"- {gate['rule']}",
            "",
            "## Traces",
            "",
            "- `results/e2_timeout_poc_control_trace.csv`",
            "- `results/e2_timeout_poc_variant_trace.csv`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--unit-tests-only", action="store_true")
    args = parser.parse_args()

    print("E2 unit tests...")
    tests = run_e2_unit_tests()
    print(f"Unit tests passed: {tests}")
    if args.unit_tests_only:
        return

    protocol = load_protocol()
    cfg = protocol["e2_timeout_poc"]
    n_eval = args.eval_episodes or cfg.get("eval_episodes", 100)
    seed_base = cfg.get("eval_seed_base", 99000)
    ckpt = PROJECT_ROOT / "models" / "checkpoints"
    control_path = ckpt / "masked_ppo_multifault_control_star.zip"
    e2_path = ckpt / "masked_ppo_multifault_e2_timeout_star.zip"

    if control_path.with_suffix(".zip").exists() and not getattr(args, "retrain_control", False):
        print("Loading existing control...")
        control_model = MaskablePPO.load(str(control_path.with_suffix("")))
    else:
        print("Training control...")
        set_seeds(cfg["policy_seed"])
        env_c = make_masked_base_env(cfg["topology_types"], protocol=protocol)
        check_env(env_c, warn=True)
        control_model = MaskablePPO("MultiInputPolicy", env_c, verbose=1, seed=cfg["policy_seed"])
        control_model.learn(total_timesteps=cfg["timesteps"])
        control_path.parent.mkdir(parents=True, exist_ok=True)
        control_model.save(str(control_path.with_suffix("")))
        env_c.close()

    if args.skip_train and e2_path.with_suffix(".zip").exists():
        e2_model = MaskablePPO.load(str(e2_path.with_suffix("")))
    else:
        print("Training E2 (timeout penalty)...")
        set_seeds(cfg["policy_seed"])
        env_e = make_masked_base_env(
            cfg["topology_types"], protocol=protocol, timeout_penalty=True
        )
        check_env(env_e, warn=True)
        e2_model = MaskablePPO("MultiInputPolicy", env_e, verbose=1, seed=cfg["policy_seed"])
        e2_model.learn(total_timesteps=cfg["timesteps"])
        e2_path.parent.mkdir(parents=True, exist_ok=True)
        e2_model.save(str(e2_path.with_suffix("")))
        env_e.close()

    print("Evaluating...")
    control = evaluate_model(
        control_model, protocol, n_episodes=n_eval, seed_base=seed_base, agent="masked_ppo_control"
    )
    variant = evaluate_model(
        e2_model, protocol, n_episodes=n_eval, seed_base=seed_base, agent="masked_ppo_e2_timeout"
    )
    results = PROJECT_ROOT / "results"
    write_trace_csv(results / "e2_timeout_poc_control_trace.csv", control.pop("records"))
    write_trace_csv(results / "e2_timeout_poc_variant_trace.csv", variant.pop("records"))
    gate = go_nogo(control, variant)
    report = build_report(control, variant, gate, tests, cfg)
    (PROJECT_ROOT / "reports" / "e2_timeout_poc.md").write_text(report, encoding="utf-8")
    (results / "e2_timeout_poc.json").write_text(
        json.dumps({"tests": tests, "control": control, "variant": variant, "go_nogo": gate}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"NL {control['nonlink_dsr']:.0%}→{variant['nonlink_dsr']:.0%} {gate['decision']}")
    print(f"Wrote {PROJECT_ROOT / 'reports' / 'e2_timeout_poc.md'}")


if __name__ == "__main__":
    main()
