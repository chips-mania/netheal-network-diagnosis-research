"""Stage A unit tests + Stage B Control-Obs train/probes. E4 not implemented here."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.environment.actions import DiagnosticAction, TopologyAction
from netheal.faults.injector import FaultType

from .action_mask import base_env
from .controlled_faults import inject_fault_at_location
from .env_utils import make_masked_base_env, make_masked_eval_env
from .obs_wrapper_unit_tests import run_obs_wrapper_unit_tests
from .pbrs import find_pbrs
from .protocol import PROJECT_ROOT, load_protocol
from .run_stochastic_evidence_audit import (
    _conditionals,
    _diag_probs,
    _find,
    _obs_delta,
    _p_correct,
    _tv,
)
from .run_symmetric_reward_poc import set_seeds

NONLINK = ("device_failure", "performance_degradation", "misconfiguration")
AUDIT_REF = {
    "p_phi_given_tool": 0.285,
    "p_exact_cons_given_tool": 0.007,
    "p_exact_cons_given_phi": 0.024,
}


def _evaluate_obs(model, protocol, *, n_episodes: int, seed_base: int, agent: str) -> Dict:
    from .action_trace import run_episode_with_policy
    from .run_ablation_nl_poc import _selector

    cfg = protocol["obs_then_e4_poc"]
    faults = [FaultType(ft) for ft in NONLINK]
    records: List = []
    rows: List[Dict] = []
    for i in range(n_episodes):
        fault = faults[i % len(faults)]
        env = make_masked_eval_env(
            cfg["topology_types"],
            fault_types=[fault],
            protocol=protocol,
            pbrs=True,
            pbrs_apply_shaping=False,
            evidence_obs=True,
        )
        recs, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed_base + i,
            agent=agent,
            choose_action=_selector(model, deterministic=False),
        )
        tracker = find_pbrs(env)
        bits = tracker.bits_before_diagnosis or tracker.bits.snapshot()
        phi_dx = tracker.phi_before_diagnosis
        if phi_dx is None:
            phi_dx = tracker.bits.phi
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
            }
        )
        records.extend(recs)
        env.close()
    return {"records": records, "episode_rows": rows}


def probe_one(model, protocol, *, fault, location, tool_pred, label, evidence_obs: bool) -> Dict[str, Any]:
    cfg = protocol["obs_then_e4_poc"]
    env = make_masked_eval_env(
        cfg["topology_types"],
        fault_types=[fault],
        protocol=protocol,
        pbrs=True,
        pbrs_apply_shaping=False,
        evidence_obs=evidence_obs,
    )
    obs, _ = env.reset(seed=42)
    root = base_env(env)
    inject_fault_at_location(root, fault, location)
    gt_type = root.ground_truth_fault.fault_type.value
    gt_loc = str(root.ground_truth_fault.location)
    scan = _find(root, lambda s: s.action_type == TopologyAction.SCAN_NETWORK)
    obs, _, _, _, _ = env.step(scan)
    p_scan, diag_ids = _diag_probs(model, env, obs)
    obs_scan = {k: np.array(v, copy=True) for k, v in obs.items()}
    feat0_scan = np.asarray(obs["device_status"])[:, 0].tolist()
    tool = _find(root, tool_pred)
    obs2, _, _, _, info2 = env.step(tool)
    tracker = find_pbrs(env)
    phi = int(tracker.bits.phi)
    p_tool, _ = _diag_probs(model, env, obs2)
    feat0_after = np.asarray(obs2["device_status"])[:, 0].tolist()
    payload = info2.get("action_result")
    latency = None
    if payload is not None:
        result = getattr(payload, "result", None)
        data = getattr(result, "data", None) if result is not None else None
        if isinstance(data, dict):
            latency = data.get("latency_ms") or data.get("total_latency_ms")
    p_s = _p_correct(p_scan, root, gt_type, gt_loc)
    p_t = _p_correct(p_tool, root, gt_type, gt_loc)
    env.close()
    return {
        "label": label,
        "gt_type": gt_type,
        "phi_after_tool": phi,
        "feat0_scan": feat0_scan,
        "feat0_after": feat0_after,
        "feat0_has_down": any(abs(x + 1.0) < 1e-5 for x in feat0_after),
        "max_latency_slot": float(max(np.max(obs2["device_status"][:, 4]), np.max(obs2["recent_diagnostics"][:, 4]))),
        "discovery_has_degraded": 2 in set(np.asarray(obs2["discovery_matrix"]).reshape(-1).tolist()),
        "tool_latency_ms": latency,
        "tv_diagnosis": _tv(p_scan, p_tool, diag_ids),
        "p_correct_scan": p_s,
        "p_correct_after_tool": p_t,
        "delta_p_correct": p_t - p_s,
        "obs_max_abs_delta": _obs_delta(obs_scan, obs2),
    }


def run_probes(model, protocol, *, evidence_obs: bool) -> List[Dict]:
    specs = [
        (
            FaultType.DEVICE_FAILURE,
            "edge_0",
            lambda s: s.action_type == DiagnosticAction.PING
            and s.parameters.get("source") == "edge_1"
            and s.parameters.get("destination") == "edge_0",
            "device_edge0_ping_to_gt",
        ),
        (
            FaultType.DEVICE_FAILURE,
            "edge_0",
            lambda s: s.action_type == DiagnosticAction.PING
            and s.parameters.get("source") == "edge_1"
            and s.parameters.get("destination") == "edge_2",
            "device_edge0_ping_healthy_pair",
        ),
        (
            FaultType.DEVICE_FAILURE,
            "edge_0",
            lambda s: s.action_type == DiagnosticAction.CHECK_STATUS
            and s.parameters.get("device") == "edge_0",
            "device_edge0_check_status",
        ),
        (
            FaultType.PERFORMANCE_DEGRADATION,
            "center->edge_0",
            lambda s: s.action_type == DiagnosticAction.PING
            and s.parameters.get("source") == "center"
            and s.parameters.get("destination") == "edge_0",
            "perf_center_edge0_ping",
        ),
        (
            FaultType.MISCONFIGURATION,
            "edge_0",
            lambda s: s.action_type == DiagnosticAction.PING
            and s.parameters.get("source") == "edge_1"
            and s.parameters.get("destination") == "edge_0",
            "misconfig_edge0_ping_to_gt",
        ),
    ]
    return [
        probe_one(model, protocol, fault=f, location=loc, tool_pred=pred, label=lab, evidence_obs=evidence_obs)
        for f, loc, pred, lab in specs
    ]


def _by_label(probes: List[Dict], name: str) -> Dict:
    return next(p for p in probes if p["label"] == name)


def decide(probes: List[Dict], sto: Dict) -> Dict:
    ev = _by_label(probes, "device_edge0_ping_to_gt")
    dummy = _by_label(probes, "device_edge0_ping_healthy_pair")
    status = _by_label(probes, "device_edge0_check_status")
    perf = _by_label(probes, "perf_center_edge0_ping")
    d_dev = max(ev["delta_p_correct"], status["delta_p_correct"])
    d_perf = perf["delta_p_correct"]
    d_dummy = dummy["delta_p_correct"]
    best = max(d_dev, d_perf)
    hypothesis = (d_dev >= 0.05 or d_perf >= 0.05) and best > d_dummy
    sto_phi = sto["p_exact_cons_given_phi"]
    sto_tool = sto["p_exact_cons_given_tool"]
    go = hypothesis and (sto_phi >= 0.10 or (best >= 0.05 and sto_tool >= 0.05))
    nogo = best < 0.02 and sto_phi <= 0.03
    if go:
        label = "go"
    elif nogo:
        label = "no_go"
    else:
        label = "ambiguous"
    return {
        "delta_p_device": d_dev,
        "delta_p_perf": d_perf,
        "delta_p_dummy": d_dummy,
        "best_delta_p": best,
        "hypothesis_support": hypothesis,
        "sto_exact_cons_given_phi": sto_phi,
        "sto_exact_cons_given_tool": sto_tool,
        "decision": label,
        "open_e4": go,
        "rule": "Go if (Device or Perf ΔP>=+0.05 and > dummy) AND (P(ec|Φ)>=10% or ΔP>=0.05 & P(ec|tool)>=5%)",
    }


def train_control_obs(protocol: dict, out_path: Path) -> MaskablePPO:
    cfg = protocol["obs_then_e4_poc"]
    set_seeds(cfg["policy_seed"])
    env = make_masked_base_env(cfg["topology_types"], protocol=protocol, evidence_obs=True)
    check_env(env, warn=True)
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, seed=cfg["policy_seed"])
    print(f"Control-Obs ent_coef={model.ent_coef} evidence_obs=True")
    model.learn(total_timesteps=cfg["timesteps"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path.with_suffix("")))
    env.close()
    return model


def build_report(tests: dict, probes: List[Dict], frozen: List[Dict], sto: Dict, gate: Dict, cfg: dict) -> str:
    def row(p):
        return (
            f"| {p['label']} | {p['phi_after_tool']} | {p['feat0_has_down']} | "
            f"{p['max_latency_slot']:.3f} | {p['delta_p_correct']:+.4f} | {p['tv_diagnosis']:.3f} |"
        )

    lines = [
        "# Stage A/B — Evidence observation wrapper + Control-Obs",
        "",
        "> NetHeal package unchanged. Reward / mask curriculum unchanged. E4 not trained.",
        f"> Train: Star 4-fault MaskablePPO {cfg['timesteps']} seed {cfg['policy_seed']}, wrapper ON.",
        "",
        "## Stage A unit tests",
        "",
        "| test | pass |",
        "|---|:---:|",
    ]
    for k, ok in tests.items():
        lines.append(f"| {k} | {ok} |")
    lines += [
        "",
        "## Stage B probes (Control-Obs, retrained)",
        "",
        "| probe | Φ | feat0 DOWN | max latency slot | ΔP(correct) | TV |",
        "|---|---:|:---:|---:|---:|---:|",
    ]
    for p in probes:
        lines.append(row(p))
    lines += [
        "",
        "## Negative control (E3 zip + wrapper, no retrain)",
        "",
        "| probe | Φ | ΔP(correct) | TV |",
        "|---|---:|---:|---:|",
    ]
    for p in frozen:
        lines.append(f"| {p['label']} | {p['phi_after_tool']} | {p['delta_p_correct']:+.4f} | {p['tv_diagnosis']:.3f} |")
    lines += [
        "",
        "## Stochastic NL (Control-Obs, n=200)",
        "",
        "| Metric | E3 zip audit | Control-Obs |",
        "|---|---:|---:|",
        f"| P(Φ≥1 \\| tool-use) | {AUDIT_REF['p_phi_given_tool']:.1%} | {sto['p_phi_given_tool']:.1%} |",
        f"| P(exact ∧ cons \\| tool-use) | {AUDIT_REF['p_exact_cons_given_tool']:.1%} | {sto['p_exact_cons_given_tool']:.1%} |",
        f"| P(exact ∧ cons \\| Φ≥1) | {AUDIT_REF['p_exact_cons_given_phi']:.1%} | {sto['p_exact_cons_given_phi']:.1%} |",
        "",
        "## Decision",
        "",
        f"- Device ΔP(correct): **{gate['delta_p_device']:+.4f}**",
        f"- Perf ΔP(correct): **{gate['delta_p_perf']:+.4f}**",
        f"- Dummy ΔP(correct): **{gate['delta_p_dummy']:+.4f}**",
        f"- Hypothesis support: **{gate['hypothesis_support']}**",
        f"- Decision: **{gate['decision']}** (open E4: {gate['open_e4']})",
        f"- {gate['rule']}",
        "",
    ]
    if not gate["open_e4"]:
        lines.append("Stage C E4 was not implemented or trained.")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--unit-tests-only", action="store_true")
    args = parser.parse_args()

    print("Stage A unit tests...")
    tests = run_obs_wrapper_unit_tests()
    print(f"Unit tests: {tests}")
    if args.unit_tests_only:
        return

    protocol = load_protocol()
    cfg = protocol["obs_then_e4_poc"]
    ckpt = PROJECT_ROOT / "models" / "checkpoints"
    obs_path = ckpt / "masked_ppo_control_obs_star.zip"
    e3_path = ckpt / "masked_ppo_e3_control_star.zip"

    if args.skip_train and obs_path.exists():
        print("Loading Control-Obs...")
        model = MaskablePPO.load(str(obs_path.with_suffix("")))
    else:
        print("Stage B: training Control-Obs...")
        model = train_control_obs(protocol, obs_path)

    print("Probes on Control-Obs...")
    probes = run_probes(model, protocol, evidence_obs=True)
    frozen = []
    if e3_path.exists():
        print("Negative control: E3 zip + wrapper...")
        frozen_model = MaskablePPO.load(str(e3_path.with_suffix("")))
        frozen = run_probes(frozen_model, protocol, evidence_obs=True)
    print("Stochastic NL eval n=200...")
    raw = _evaluate_obs(model, protocol, n_episodes=cfg["eval_episodes"], seed_base=cfg["eval_seed_base"], agent="control_obs_sto")
    sto = _conditionals(raw)
    gate = decide(probes, sto)
    report = build_report(tests, probes, frozen, sto, gate, cfg)
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "obs_then_e4_stage_b.md").write_text(report, encoding="utf-8")
    (results / "obs_then_e4_stage_b.json").write_text(
        json.dumps(
            {"unit_tests": tests, "probes": probes, "frozen_e3": frozen, "sto": sto, "decision": gate},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"decision={gate['decision']} open_e4={gate['open_e4']}")
    print(f"Wrote {reports / 'obs_then_e4_stage_b.md'}")


if __name__ == "__main__":
    main()
