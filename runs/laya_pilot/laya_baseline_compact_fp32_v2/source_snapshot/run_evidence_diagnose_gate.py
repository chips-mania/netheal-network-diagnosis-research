"""Evidence diagnose-gate: unlock diagnosis only after fault-like obs bits.

Named variant. Original reward after unlock. Dummy healthy ping does not unlock.
Control = existing Control-Obs zip, same eval protocol, no gate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_trace import aggregate_behavior, run_episode_with_policy
from .diagnosis_decomposition import aggregate_decomposition
from .env_utils import make_masked_base_env, make_masked_eval_env, metrics_env
from .evidence_diagnose_gate import smoke_test
from .metrics_utils import summarize_evaluator
from .pbrs import find_pbrs
from .protocol import PROJECT_ROOT, load_protocol
from .run_ablation_nl_poc import HARD, NONLINK, _per_fault, _row_consistent, _selector
from .run_diagnosis_gate_poc import shortcut_rate
from .run_symmetric_reward_poc import set_seeds
from .train_episode_log import find_train_log, summarize_rows


def _evaluate(model, protocol, *, gate: bool, n: int, seed_base: int, det: bool) -> Dict:
    cfg = protocol["evidence_diagnose_gate_poc"]
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    evaluator = CompetitionEvaluator()
    records: List = []
    summaries: List = []
    rows: List[Dict] = []
    for i in range(n):
        fault = faults[i % len(faults)]
        env = make_masked_eval_env(
            cfg["topology_types"],
            fault_types=[fault],
            protocol=protocol,
            evidence_obs=True,
            evidence_diagnose_gate=gate,
            pbrs=True,
            pbrs_apply_shaping=False,
        )
        recs, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed_base + i,
            agent="gate" if gate else "control_obs",
            choose_action=_selector(model, deterministic=det),
        )
        m = metrics_env(env).last_episode_metrics
        if m:
            evaluator.add_episode_metrics(m)
        tracker = find_pbrs(env)
        bits = tracker.bits_before_diagnosis or tracker.bits.snapshot()
        phi = tracker.phi_before_diagnosis
        if phi is None:
            phi = tracker.bits.phi
        rows.append(
            {
                "fault_type": summary.fault_type,
                "fault_location": summary.fault_location,
                "success": summary.success,
                "predicted_type": summary.predicted_type,
                "predicted_location": summary.predicted_location,
                "phi_before_dx": int(phi),
                "type_bit": int(bits.get("type_bit", 0)),
                "loc_bit": int(bits.get("loc_bit", 0)),
                "steps": summary.steps,
                "submitted": summary.predicted_type is not None,
            }
        )
        records.extend(recs)
        summaries.append(summary)
        env.close()

    timeout = sum(1 for s in summaries if (not s.success) and s.steps >= protocol["max_episode_steps"])
    submitted = sum(1 for r in rows if r["submitted"])
    exact_cons = [r for r in rows if r["success"] and _row_consistent(r)]
    return {
        "stats": summarize_evaluator(evaluator),
        "behavior": aggregate_behavior(summaries, records),
        "decomposition": aggregate_decomposition(summaries),
        "per_fault_dsr": aggregate_behavior(summaries, records).get("fault_type_dsr", {}),
        "per_fault": _per_fault(rows, summaries),
        "shortcut": shortcut_rate(summaries, records),
        "timeout_rate": timeout / max(1, n),
        "submit_rate": submitted / max(1, n),
        "exact_consistent_n": len(exact_cons),
        "n": n,
    }


def _go(control: Dict, variant: Dict) -> Dict:
    sc = variant["shortcut"]["shortcut_rate"]
    timeout = variant["timeout_rate"]
    hard_ok = False
    for ft in HARD:
        b = variant["per_fault"].get(ft, {})
        if int(b.get("exact_n", 0)) >= 2 and int(b.get("exact_consistent_n", 0)) >= 2:
            hard_ok = True
    avoidance = timeout >= 0.70 or variant["submit_rate"] < 0.20
    dummy = (
        sc < 0.10
        and variant["stats"]["avg_steps_all"] <= 3.2
        and variant["behavior"]["avg_tool_actions_before_diagnosis"] <= 2.2
        and variant["exact_consistent_n"] == 0
    )
    go = sc < 0.10 and hard_ok and not avoidance and not dummy
    nogo = avoidance or dummy or (not hard_ok and variant["exact_consistent_n"] == 0)
    if go:
        label = "go"
    elif nogo:
        label = "no_go"
    else:
        label = "ambiguous"
    return {
        "decision": label,
        "shortcut": sc,
        "timeout_rate": timeout,
        "submit_rate": variant["submit_rate"],
        "hard_ok": hard_ok,
        "dummy_like": dummy,
        "avoidance": avoidance,
        "exact_consistent_n": variant["exact_consistent_n"],
        "rule": "Go if shortcut<10% and Device or Misconfig exact>=2 & cons>=2, not timeout/dummy",
    }


def main() -> None:
    protocol = load_protocol()
    cfg = protocol["evidence_diagnose_gate_poc"]
    smoke = smoke_test()
    print("smoke", smoke)
    if not all(smoke.values()):
        raise SystemExit(f"smoke failed: {smoke}")

    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_evidence_gate_star.zip"
    log_path = results / "train_episode_log" / "evidence_gate_star.jsonl"
    control_zip = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_control_obs_star.zip"

    set_seeds(cfg["policy_seed"])
    env = make_masked_base_env(
        cfg["topology_types"],
        protocol=protocol,
        evidence_obs=True,
        evidence_diagnose_gate=True,
        episode_log_path=log_path,
    )
    check_env(env, warn=True)
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, seed=cfg["policy_seed"])
    print("EvidenceDiagnoseGate train 30k evidence_obs=True original reward")
    model.learn(total_timesteps=cfg["timesteps"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt.with_suffix("")))
    logger = find_train_log(env)
    train_sum = summarize_rows(logger.rows) if logger else {}
    env.close()

    n = cfg["eval_episodes"]
    seed_base = cfg["eval_seed_base"]
    print("Eval variant det...")
    variant = _evaluate(model, protocol, gate=True, n=n, seed_base=seed_base, det=True)
    control = None
    if control_zip.exists():
        print("Eval Control-Obs det (no gate)...")
        ctrl = MaskablePPO.load(str(control_zip))
        control = _evaluate(ctrl, protocol, gate=False, n=n, seed_base=seed_base, det=True)
    gate = _go(control or {}, variant)

    def _pf(block, ft):
        b = block.get("per_fault", {}).get(ft, {})
        return f"{b.get('dsr', 0):.0%} ({b.get('exact_n', 0)}/{b.get('n', 0)}; cons {b.get('exact_consistent_n', 0)})"

    lines = [
        "# Evidence diagnose-gate PoC",
        "",
        "> Diagnosis locked until fault-like bits in obs (DOWN / latency≥15ms / discovery=2 / failed tool).",
        "> Healthy dummy ping does not unlock. Reward after unlock is original. Named variant.",
        f"> Smoke: {smoke}",
        "",
        "| Metric | Control-Obs | Evidence-gate |",
        "|---|---:|---:|",
    ]
    if control:
        lines += [
            f"| Exact DSR | {control['decomposition']['exact_diagnosis_accuracy']:.0%} | {variant['decomposition']['exact_diagnosis_accuracy']:.0%} |",
            f"| Shortcut | {control['shortcut']['shortcut_rate']:.0%} | {variant['shortcut']['shortcut_rate']:.0%} |",
            f"| AvgSteps | {control['stats']['avg_steps_all']:.1f} | {variant['stats']['avg_steps_all']:.1f} |",
            f"| Timeout | {control['timeout_rate']:.0%} | {variant['timeout_rate']:.0%} |",
            f"| Submit | {control['submit_rate']:.0%} | {variant['submit_rate']:.0%} |",
            f"| Device | {_pf(control, 'device_failure')} | {_pf(variant, 'device_failure')} |",
            f"| Perf | {_pf(control, 'performance_degradation')} | {_pf(variant, 'performance_degradation')} |",
            f"| Misconfig | {_pf(control, 'misconfiguration')} | {_pf(variant, 'misconfiguration')} |",
            f"| Link | {_pf(control, 'link_failure')} | {_pf(variant, 'link_failure')} |",
        ]
    else:
        lines.append("| Control-Obs zip missing | | |")
    lines += [
        "",
        f"- Train-time exact∧cons\\|tool: {train_sum.get('p_exact_cons_given_tool', 0):.1%}",
        f"- Decision: **{gate['decision']}**",
        f"- {gate['rule']}",
        "",
    ]
    (reports / "evidence_diagnose_gate_poc.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {"smoke": smoke, "train": train_sum, "variant": {k: v for k, v in variant.items() if k != "records"}, "gate": gate}
    if control:
        payload["control"] = {k: v for k, v in control.items() if k != "records"}
    # drop non-serializable
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("records",)}
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                return str(o)
        return o

    (results / "evidence_diagnose_gate_poc.json").write_text(
        json.dumps(_clean(payload), indent=2, default=str), encoding="utf-8"
    )
    print(f"decision={gate['decision']} shortcut={gate['shortcut']:.0%} timeout={gate['timeout_rate']:.0%}")
    print("Wrote", reports / "evidence_diagnose_gate_poc.md")


if __name__ == "__main__":
    main()
