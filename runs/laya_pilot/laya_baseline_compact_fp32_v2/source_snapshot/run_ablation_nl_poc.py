"""Ablation-NL — train without link GT; keep 72-way action space."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.environment.actions import ActionCategory
from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_mask import action_masks_for_env
from .action_trace import aggregate_behavior, run_episode_with_policy, write_trace_csv
from .diagnosis_decomposition import aggregate_decomposition
from .env_utils import make_masked_base_env, make_masked_eval_env, metrics_env
from .metrics_utils import summarize_evaluator
from .pbrs import find_pbrs
from .protocol import PROJECT_ROOT, load_protocol
from .run_diagnosis_gate_poc import shortcut_rate
from .run_e1_pbrs_poc import evidence_from_episodes
from .run_symmetric_reward_poc import set_seeds

NONLINK = ("device_failure", "performance_degradation", "misconfiguration")
HARD = ("device_failure", "misconfiguration")
ALL_FAULTS = (
    "link_failure",
    "device_failure",
    "performance_degradation",
    "misconfiguration",
)


def _selector(model: MaskablePPO, *, deterministic: bool):
    def choose(env, obs, _step, _records):
        masks = action_masks_for_env(env)
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
        return int(action)

    return choose


def _row_consistent(r: Dict) -> bool:
    if int(r.get("phi_before_dx", 0)) < 1:
        return False
    if r.get("type_bit") and r.get("predicted_type") != r.get("fault_type"):
        return False
    from .diagnosis_decomposition import location_match

    if r.get("loc_bit") and not location_match(
        r.get("fault_location") or "", r.get("predicted_location") or "", r.get("fault_type") or ""
    ):
        return False
    return True


def _per_fault(rows: List[Dict], summaries) -> Dict[str, Dict]:
    by_ft: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_ft[r["fault_type"]].append(r)
    out = {}
    for ft, sub in by_ft.items():
        n = len(sub)
        exact = [r for r in sub if r["success"]]
        exact_cons = [r for r in exact if _row_consistent(r)]
        acquired = [r for r in sub if int(r.get("phi_before_dx", 0)) >= 1]
        out[ft] = {
            "n": n,
            "exact_n": len(exact),
            "exact_consistent_n": len(exact_cons),
            "dsr": len(exact) / max(1, n),
            "acquisition": len(acquired) / max(1, n),
        }
    for s in summaries:
        out.setdefault(s.fault_type, {"n": 0, "exact_n": 0, "exact_consistent_n": 0, "dsr": 0.0, "acquisition": 0.0})
    return out


def evaluate(
    model: MaskablePPO,
    protocol: dict,
    *,
    fault_names: Sequence[str],
    n_episodes: int,
    seed_base: int,
    deterministic: bool,
    agent: str,
) -> Dict:
    cfg = protocol["ablation_nl_poc"]
    faults = [FaultType(ft) for ft in fault_names]
    evaluator = CompetitionEvaluator()
    records: List = []
    summaries: List = []
    rows: List[Dict] = []

    for i in range(n_episodes):
        fault = faults[i % len(faults)]
        env = make_masked_eval_env(
            cfg["topology_types"],
            fault_types=[fault],
            protocol=protocol,
            pbrs=True,
            pbrs_apply_shaping=False,
        )
        recs, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed_base + i,
            agent=agent,
            choose_action=_selector(model, deterministic=deterministic),
        )
        m = metrics_env(env).last_episode_metrics
        if m:
            evaluator.add_episode_metrics(m)
        tracker = find_pbrs(env)
        bits = tracker.bits_before_diagnosis or tracker.bits.snapshot()
        phi_dx = tracker.phi_before_diagnosis
        if phi_dx is None:
            phi_dx = tracker.bits.phi
        first_phi = next((r.step for r in recs if r.action_category == "diagnostic"), None)
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
        records.extend(recs)
        summaries.append(summary)
        env.close()

    nl = [s for s in summaries if s.fault_type in NONLINK]
    diag_before = (
        float(
            np.mean(
                [
                    any(
                        r.episode_id == s.episode_id and r.action_category == ActionCategory.DIAGNOSTIC.value
                        for r in records
                    )
                    for s in nl
                ]
            )
        )
        if nl
        else 0.0
    )
    ev = evidence_from_episodes(rows)
    decomp = aggregate_decomposition(summaries)
    behavior = aggregate_behavior(summaries, records)
    return {
        "deterministic": deterministic,
        "fault_names": list(fault_names),
        "stats": summarize_evaluator(evaluator),
        "behavior": behavior,
        "decomposition": {
            "fault_type_accuracy": decomp["fault_type_accuracy"],
            "location_accuracy": decomp["location_accuracy"],
            "exact_diagnosis_accuracy": decomp["exact_diagnosis_accuracy"],
        },
        "per_fault_dsr": behavior.get("fault_type_dsr", {}),
        "per_fault": _per_fault(rows, summaries),
        "shortcut": shortcut_rate(summaries, records),
        "avg_steps": float(np.mean([s.steps for s in summaries])) if summaries else 0.0,
        "tool_use_before_dx": diag_before,
        "evidence": ev,
        "records": records,
    }


def train_nl(protocol: dict, out_path: Path) -> MaskablePPO:
    cfg = protocol["ablation_nl_poc"]
    set_seeds(cfg["policy_seed"])
    fault_types = [FaultType(ft) for ft in cfg["train_fault_types"]]
    env = make_masked_base_env(cfg["topology_types"], fault_types=fault_types, protocol=protocol)
    check_env(env, warn=True)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=cfg["policy_seed"],
    )
    print(f"train fault_types={[ft.value for ft in fault_types]} ent_coef={model.ent_coef}")
    model.learn(total_timesteps=cfg["timesteps"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path.with_suffix("")))
    env.close()
    return model


def train_control(protocol: dict, out_path: Path) -> MaskablePPO:
    cfg = protocol["ablation_nl_poc"]
    set_seeds(cfg["policy_seed"])
    env = make_masked_base_env(cfg["topology_types"], protocol=protocol)
    check_env(env, warn=True)
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, seed=cfg["policy_seed"])
    model.learn(total_timesteps=cfg["timesteps"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path.with_suffix("")))
    env.close()
    return model


def decide(c_det: Dict, v_det: Dict) -> Dict:
    support_faults = []
    for ft in HARD:
        vf = v_det["per_fault"].get(ft, {})
        if int(vf.get("exact_n", 0)) >= 2 and int(vf.get("exact_consistent_n", 0)) >= 2:
            support_faults.append(ft)
    hypothesis = bool(support_faults)
    c_nl = c_det["evidence"]["nonlink_dsr"]
    v_nl = v_det["evidence"]["nonlink_dsr"]
    d_nl = v_nl - c_nl
    go = hypothesis and d_nl >= 0.05
    c_sc = c_det["shortcut"]["shortcut_rate"]
    v_sc = v_det["shortcut"]["shortcut_rate"]
    shortcut_same = abs(v_sc - c_sc) < 0.05 and abs(v_det["avg_steps"] - c_det["avg_steps"]) < 0.3
    perf_up = v_det["per_fault"].get("performance_degradation", {}).get("dsr", 0) > (
        c_det["per_fault"].get("performance_degradation", {}).get("dsr", 0) + 0.02
    )
    if go:
        label = "go"
    elif hypothesis:
        label = "hypothesis_support_no_seed"
    elif perf_up:
        label = "ambiguous"
    elif shortcut_same:
        label = "no_go"
    else:
        label = "no_go"
    return {
        "hypothesis_support": hypothesis,
        "support_faults": support_faults,
        "nonlink_dsr_delta": d_nl,
        "go": go,
        "decision": label,
        "shortcut_same": shortcut_same,
        "perf_up": perf_up,
        "rule": "Go if Device/Misconfig det exact>=2 and exact-consistent>=2 AND NL DSR +5%p (det)",
    }


def _fmt_ft(block: Dict, ft: str) -> str:
    b = block.get(ft, {})
    return f"{b.get('dsr', 0):.0%} ({b.get('exact_n', 0)}/{b.get('n', 0)}; cons {b.get('exact_consistent_n', 0)})"


def build_report(c_sto, c_det, v_sto, v_det, v4_sto, v4_det, c4_det, gate, cfg, control_src: str) -> str:
    lines = [
        "# Ablation-NL — Non-link-only train distribution (seed 1)",
        "",
        "> Reward / 72-way action space unchanged. GT sampler excludes link. E1/E2/E3 not mixed.",
        "> Primary eval = non-link-only, same n/seeds for Control and Ablation-NL.",
        "> Go/No-Go uses **deterministic** primary. Stochastic is recorded.",
        "",
        f"Train: Star, 6 devices, MaskablePPO, {cfg['timesteps']} timesteps, seed {cfg['policy_seed']}",
        f"Control weights: {control_src}",
        "Confound: ~1.33× non-link exposure (timesteps, not episodes). Not separated from competition removal.",
        "",
        "## Primary — non-link-only eval",
        "",
        "| Metric | Control sto | Ablation-NL sto | Control det | Ablation-NL det |",
        "|---|---:|---:|---:|---:|",
        f"| Non-link DSR | {c_sto['evidence']['nonlink_dsr']:.0%} | {v_sto['evidence']['nonlink_dsr']:.0%} | "
        f"{c_det['evidence']['nonlink_dsr']:.0%} | {v_det['evidence']['nonlink_dsr']:.0%} |",
        f"| Device | {_fmt_ft(c_sto['per_fault'], 'device_failure')} | {_fmt_ft(v_sto['per_fault'], 'device_failure')} | "
        f"{_fmt_ft(c_det['per_fault'], 'device_failure')} | {_fmt_ft(v_det['per_fault'], 'device_failure')} |",
        f"| Perf | {_fmt_ft(c_sto['per_fault'], 'performance_degradation')} | {_fmt_ft(v_sto['per_fault'], 'performance_degradation')} | "
        f"{_fmt_ft(c_det['per_fault'], 'performance_degradation')} | {_fmt_ft(v_det['per_fault'], 'performance_degradation')} |",
        f"| Misconfig | {_fmt_ft(c_sto['per_fault'], 'misconfiguration')} | {_fmt_ft(v_sto['per_fault'], 'misconfiguration')} | "
        f"{_fmt_ft(c_det['per_fault'], 'misconfiguration')} | {_fmt_ft(v_det['per_fault'], 'misconfiguration')} |",
        f"| Tool-use (pre-dx) | {c_sto['tool_use_before_dx']:.0%} | {v_sto['tool_use_before_dx']:.0%} | "
        f"{c_det['tool_use_before_dx']:.0%} | {v_det['tool_use_before_dx']:.0%} |",
        f"| Evidence Acquisition | {c_sto['evidence']['evidence_acquisition']:.0%} | {v_sto['evidence']['evidence_acquisition']:.0%} | "
        f"{c_det['evidence']['evidence_acquisition']:.0%} | {v_det['evidence']['evidence_acquisition']:.0%} |",
        f"| Evidence-Consistent | {c_sto['evidence']['evidence_consistency']:.0%} | {v_sto['evidence']['evidence_consistency']:.0%} | "
        f"{c_det['evidence']['evidence_consistency']:.0%} | {v_det['evidence']['evidence_consistency']:.0%} |",
        f"| Shortcut | {c_sto['shortcut']['shortcut_rate']:.0%} | {v_sto['shortcut']['shortcut_rate']:.0%} | "
        f"{c_det['shortcut']['shortcut_rate']:.0%} | {v_det['shortcut']['shortcut_rate']:.0%} |",
        f"| AvgSteps | {c_sto['avg_steps']:.2f} | {v_sto['avg_steps']:.2f} | {c_det['avg_steps']:.2f} | {v_det['avg_steps']:.2f} |",
        "",
        "## Secondary — Ablation-NL on 4-fault mix (reference)",
        "",
        f"- Link DSR sto/det: {v4_sto['per_fault_dsr'].get('link_failure', 0):.0%} / {v4_det['per_fault_dsr'].get('link_failure', 0):.0%}",
        f"- 4-fault Exact DSR sto/det: {v4_sto['decomposition']['exact_diagnosis_accuracy']:.0%} / {v4_det['decomposition']['exact_diagnosis_accuracy']:.0%}",
        f"- Control 4-fault det Exact (zip check): {c4_det['decomposition']['exact_diagnosis_accuracy']:.0%} (E3 published 20%)",
        "",
        "## Decision (deterministic NL eval)",
        "",
        f"- Hypothesis support: **{gate['hypothesis_support']}** faults={gate['support_faults']}",
        f"- Non-link DSR Δ: **{gate['nonlink_dsr_delta']:+.1%}** (need ≥ +5%p for Go)",
        f"- Shortcut/AvgSteps similar: {gate['shortcut_same']}",
        f"- Perf-only up: {gate['perf_up']}",
        f"- Decision: **{gate['decision']}**",
        f"- {gate['rule']}",
        "",
    ]
    return "\n".join(lines)


def _strip(d: Dict) -> Dict:
    d = dict(d)
    d.pop("records", None)
    d.pop("behavior", None)
    return d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()

    protocol = load_protocol()
    cfg = protocol["ablation_nl_poc"]
    n_eval = args.eval_episodes or cfg.get("eval_episodes", 100)
    seed_det = cfg.get("eval_seed_base", 101000)
    seed_stoch = seed_det + 5000
    seed_4 = seed_det + 10000
    ckpt = PROJECT_ROOT / "models" / "checkpoints"
    e3_control = ckpt / "masked_ppo_e3_control_star.zip"
    nl_path = ckpt / "masked_ppo_ablation_nl_star.zip"
    fallback_control = ckpt / "masked_ppo_ablation_nl_control_star.zip"

    if e3_control.exists():
        print(f"Loading E3 Control weights: {e3_control}")
        control = MaskablePPO.load(str(e3_control.with_suffix("")))
        control_src = str(e3_control.name)
    elif args.skip_train and fallback_control.exists():
        print("Loading fallback Control...")
        control = MaskablePPO.load(str(fallback_control.with_suffix("")))
        control_src = str(fallback_control.name)
    else:
        print("E3 Control zip missing — training 4-fault Control...")
        control = train_control(protocol, fallback_control)
        control_src = str(fallback_control.name) + " (retrained)"

    if args.skip_train and nl_path.exists():
        print("Loading Ablation-NL...")
        variant = MaskablePPO.load(str(nl_path.with_suffix("")))
    else:
        print("Training Ablation-NL (no link GT)...")
        variant = train_nl(protocol, nl_path)

    print("Primary NL eval (sto + det)...")
    c_sto = evaluate(
        control, protocol, fault_names=NONLINK, n_episodes=n_eval, seed_base=seed_stoch, deterministic=False, agent="control_nl_sto"
    )
    v_sto = evaluate(
        variant, protocol, fault_names=NONLINK, n_episodes=n_eval, seed_base=seed_stoch, deterministic=False, agent="nl_sto"
    )
    c_det = evaluate(
        control, protocol, fault_names=NONLINK, n_episodes=n_eval, seed_base=seed_det, deterministic=True, agent="control_nl_det"
    )
    v_det = evaluate(
        variant, protocol, fault_names=NONLINK, n_episodes=n_eval, seed_base=seed_det, deterministic=True, agent="nl_det"
    )
    print("Secondary 4-fault eval...")
    v4_sto = evaluate(
        variant, protocol, fault_names=ALL_FAULTS, n_episodes=n_eval, seed_base=seed_4, deterministic=False, agent="nl_4_sto"
    )
    v4_det = evaluate(
        variant, protocol, fault_names=ALL_FAULTS, n_episodes=n_eval, seed_base=seed_4, deterministic=True, agent="nl_4_det"
    )
    c4_det = evaluate(
        control, protocol, fault_names=ALL_FAULTS, n_episodes=n_eval, seed_base=seed_4, deterministic=True, agent="control_4_det"
    )

    results = PROJECT_ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    write_trace_csv(results / "ablation_nl_control_nl_det_trace.csv", c_det["records"])
    write_trace_csv(results / "ablation_nl_variant_nl_det_trace.csv", v_det["records"])
    write_trace_csv(results / "ablation_nl_variant_4fault_det_trace.csv", v4_det["records"])

    gate = decide(c_det, v_det)
    report = build_report(
        _strip(c_sto), _strip(c_det), _strip(v_sto), _strip(v_det),
        _strip(v4_sto), _strip(v4_det), _strip(c4_det),
        gate, cfg, control_src,
    )
    (PROJECT_ROOT / "reports" / "ablation_nl_poc.md").write_text(report, encoding="utf-8")
    (results / "ablation_nl_poc.json").write_text(
        json.dumps(
            {
                "control_src": control_src,
                "control_nl_sto": _strip(c_sto),
                "control_nl_det": _strip(c_det),
                "nl_sto": _strip(v_sto),
                "nl_det": _strip(v_det),
                "nl_4_sto": _strip(v4_sto),
                "nl_4_det": _strip(v4_det),
                "control_4_det": _strip(c4_det),
                "decision": gate,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"decision={gate['decision']} ΔNL={gate['nonlink_dsr_delta']:+.1%}")
    print(f"Wrote {PROJECT_ROOT / 'reports' / 'ablation_nl_poc.md'}")


if __name__ == "__main__":
    main()
