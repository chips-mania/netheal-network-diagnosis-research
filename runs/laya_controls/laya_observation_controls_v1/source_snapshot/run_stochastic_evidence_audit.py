"""Pre-E4 audit: tool-use-conditional evidence + obs/diagnose conditioning."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sb3_contrib import MaskablePPO

from netheal.environment.actions import ActionCategory, DiagnosticAction, TopologyAction
from netheal.faults.injector import FaultType

from .action_mask import action_masks_for_env, base_env
from .controlled_faults import inject_fault_at_location
from .env_utils import make_masked_eval_env
from .pbrs import find_pbrs
from .protocol import PROJECT_ROOT, load_protocol
from .run_ablation_nl_poc import NONLINK, _row_consistent, evaluate

N_STOCH = 400
SEED_BASE = 202600


def _find(root, pred) -> int:
    for aid, spec in root.action_space_manager.action_map.items():
        if pred(spec):
            return aid
    raise RuntimeError("action not found")


def _conditionals(result: Dict) -> Dict[str, Any]:
    records = result["records"]
    # Rebuild episode flags from evaluate() rows via per-episode records.
    # evaluate() does not keep rows; recompute tool-use and join on episode_id from traces.
    by_ep: Dict[int, List] = defaultdict(list)
    for rec in records:
        by_ep[rec.episode_id].append(rec)

    episodes = []
    for eid, recs in by_ep.items():
        used_tool = any(r.action_category == ActionCategory.DIAGNOSTIC.value for r in recs)
        last = recs[-1]
        episodes.append(
            {
                "episode_id": eid,
                "fault_type": recs[0].fault_type,
                "tool_use": used_tool,
                "exact": bool(last.diagnosis_correct),
            }
        )

    # Rich evidence is not in StepRecord; re-read from sidecar if present.
    ev_rows = result.get("episode_rows") or []
    if ev_rows:
        ev_by = {i: r for i, r in enumerate(ev_rows)}
        for ep in episodes:
            r = ev_by.get(ep["episode_id"], {})
            ep["phi"] = int(r.get("phi_before_dx", 0))
            ep["exact"] = bool(r.get("success", ep["exact"]))
            ep["consistent"] = _row_consistent(r) if r else False
            ep["exact_consistent"] = bool(ep["exact"] and ep["consistent"])
    else:
        for ep in episodes:
            ep["phi"] = 0
            ep["consistent"] = False
            ep["exact_consistent"] = False

    n = max(1, len(episodes))
    tool = [e for e in episodes if e["tool_use"]]
    acquired = [e for e in episodes if e.get("phi", 0) >= 1]
    nt = max(1, len(tool))
    na = max(1, len(acquired))
    by_ft = defaultdict(lambda: {"n": 0, "tool": 0, "phi": 0, "exact_cons": 0})
    for e in episodes:
        b = by_ft[e["fault_type"]]
        b["n"] += 1
        b["tool"] += int(e["tool_use"])
        b["phi"] += int(e.get("phi", 0) >= 1)
        b["exact_cons"] += int(e.get("exact_consistent", False))
    return {
        "n": len(episodes),
        "n_tool_use": len(tool),
        "n_phi": len(acquired),
        "p_tool_use": len(tool) / n,
        "p_phi": len(acquired) / n,
        "p_phi_given_tool": sum(e.get("phi", 0) >= 1 for e in tool) / nt,
        "p_exact_cons_given_tool": sum(e.get("exact_consistent") for e in tool) / nt,
        "p_exact_cons_given_phi": sum(e.get("exact_consistent") for e in acquired) / na,
        "n_exact_consistent": sum(e.get("exact_consistent") for e in episodes),
        "per_fault": dict(by_ft),
    }


def _patch_evaluate_rows():
    """Wrap evaluate to keep episode_rows for conditionals."""
    from . import run_ablation_nl_poc as m

    orig = m.evaluate

    def wrapped(*args, **kwargs):
        # Call a local copy that stores rows.
        return _evaluate_with_rows(*args, **kwargs)

    return wrapped


def _evaluate_with_rows(model, protocol, *, fault_names, n_episodes, seed_base, deterministic, agent):
    from .action_trace import run_episode_with_policy
    from .env_utils import make_masked_eval_env, metrics_env
    from .pbrs import find_pbrs
    from .run_ablation_nl_poc import _selector

    cfg = protocol["ablation_nl_poc"]
    faults = [FaultType(ft) for ft in fault_names]
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
        )
        recs, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed_base + i,
            agent=agent,
            choose_action=_selector(model, deterministic=deterministic),
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


def _obs_delta(a: dict, b: dict) -> Dict[str, float]:
    out = {}
    for k in a:
        va, vb = np.asarray(a[k], dtype=np.float64), np.asarray(b[k], dtype=np.float64)
        out[k] = float(np.max(np.abs(va - vb))) if va.shape == vb.shape else -1.0
    return out


def _diag_probs(model: MaskablePPO, env, obs) -> Tuple[np.ndarray, List[int]]:
    import torch

    root = base_env(env)
    masks = action_masks_for_env(env)
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(
            obs_t, action_masks=torch.as_tensor(np.array([masks]))
        )
        probs = dist.distribution.probs.cpu().numpy().flatten()
    diag_ids = [
        aid
        for aid, spec in root.action_space_manager.action_map.items()
        if spec.category == ActionCategory.DIAGNOSIS and masks[aid]
    ]
    return probs, diag_ids


def _tv(p: np.ndarray, q: np.ndarray, ids: List[int]) -> float:
    return 0.5 * float(np.abs(p[ids] - q[ids]).sum())


def _p_correct(probs: np.ndarray, root, gt_type: str, gt_loc: str) -> float:
    total = 0.0
    for aid, spec in root.action_space_manager.action_map.items():
        if spec.category != ActionCategory.DIAGNOSIS:
            continue
        ft = spec.action_type.value if hasattr(spec.action_type, "value") else str(spec.action_type)
        loc = str(spec.parameters.get("location", ""))
        if ft != gt_type:
            continue
        if gt_type in ("link_failure", "performance_degradation"):
            if sorted(p.strip() for p in loc.split("->")) == sorted(p.strip() for p in gt_loc.split("->")):
                total += float(probs[aid])
        elif loc == gt_loc:
            total += float(probs[aid])
    return total


def _probe_one(
    model: MaskablePPO,
    protocol: dict,
    *,
    fault: FaultType,
    location: str,
    tool_pred,
    label: str,
) -> Dict[str, Any]:
    env = make_masked_eval_env(
        protocol["ablation_nl_poc"]["topology_types"],
        fault_types=[fault],
        protocol=protocol,
        pbrs=True,
        pbrs_apply_shaping=False,
    )
    obs, info = env.reset(seed=42)
    root = base_env(env)
    inject_fault_at_location(root, fault, location)
    gt_type = root.ground_truth_fault.fault_type.value
    gt_loc = str(root.ground_truth_fault.location)
    scan = _find(root, lambda s: s.action_type == TopologyAction.SCAN_NETWORK)
    obs, _, _, _, _ = env.step(scan)
    p_scan, diag_ids = _diag_probs(model, env, obs)
    obs_scan = {k: np.array(v, copy=True) for k, v in obs.items()}
    raw_status_scan = {
        did: str(getattr(info.get("status"), "name", info.get("status")))
        for did, info in root.observation_manager.device_status.devices.items()
    } if hasattr(root, "observation_manager") else {}
    # StructuredObservation lives on env
    struct = getattr(root, "structured_obs", None) or getattr(root, "observation", None)
    status_scan = {}
    if struct is not None and hasattr(struct, "device_status"):
        status_scan = {
            did: str(info.get("status"))
            for did, info in struct.device_status.devices.items()
        }
    device_feat0_scan = np.asarray(obs["device_status"])[:, 0].tolist() if "device_status" in obs else []

    tool = _find(root, tool_pred)
    obs2, _, _, _, info2 = env.step(tool)
    tracker = find_pbrs(env)
    phi = int(tracker.bits.phi)
    p_tool, _ = _diag_probs(model, env, obs2)
    delta = _obs_delta(obs_scan, obs2)
    status_after = {}
    if struct is not None and hasattr(struct, "device_status"):
        status_after = {
            did: str(info.get("status"))
            for did, info in struct.device_status.devices.items()
        }
    conn_props = {}
    if struct is not None and hasattr(struct, "discovery_matrix"):
        conn_props = {
            str(k): dict(v) for k, v in struct.discovery_matrix.connection_properties.items()
        }
    device_feat0_after = np.asarray(obs2["device_status"])[:, 0].tolist()
    payload = info2.get("action_result")
    tool_ok = None
    latency = None
    err = None
    if payload is not None:
        result = getattr(payload, "result", None)
        data = getattr(result, "data", None) if result is not None else None
        tool_ok = getattr(result, "success", None) if result is not None else None
        if isinstance(data, dict):
            latency = data.get("latency_ms") or data.get("total_latency_ms")
            err = data.get("error")
    env.close()
    return {
        "label": label,
        "gt_type": gt_type,
        "gt_loc": gt_loc,
        "phi_after_tool": phi,
        "obs_max_abs_delta": delta,
        "device_status_feat0_changed": device_feat0_scan != device_feat0_after,
        "device_status_feat0_scan": device_feat0_scan,
        "device_status_feat0_after": device_feat0_after,
        "internal_status_scan": status_scan,
        "internal_status_after": status_after,
        "connection_properties_in_obs": False,
        "n_connection_properties_internal": len(conn_props),
        "tool_success": tool_ok,
        "tool_latency_ms": latency,
        "tool_error": err,
        "tv_diagnosis": _tv(p_scan, p_tool, diag_ids),
        "p_correct_scan": _p_correct(p_scan, root, gt_type, gt_loc),
        "p_correct_after_tool": _p_correct(p_tool, root, gt_type, gt_loc),
        "argmax_scan": int(np.argmax(p_scan)),
        "argmax_after": int(np.argmax(p_tool)),
    }


def run_conditioning_probes(model: MaskablePPO, protocol: dict) -> List[Dict]:
    probes = []
    # Evidence ping: device down at edge_0
    probes.append(
        _probe_one(
            model,
            protocol,
            fault=FaultType.DEVICE_FAILURE,
            location="edge_0",
            tool_pred=lambda s: (
                s.action_type == DiagnosticAction.PING
                and s.parameters.get("source") == "edge_1"
                and s.parameters.get("destination") == "edge_0"
            ),
            label="device_edge0_ping_to_gt",
        )
    )
    # Dummy ping: healthy pair
    probes.append(
        _probe_one(
            model,
            protocol,
            fault=FaultType.DEVICE_FAILURE,
            location="edge_0",
            tool_pred=lambda s: (
                s.action_type == DiagnosticAction.PING
                and s.parameters.get("source") == "edge_1"
                and s.parameters.get("destination") == "edge_2"
            ),
            label="device_edge0_ping_healthy_pair",
        )
    )
    probes.append(
        _probe_one(
            model,
            protocol,
            fault=FaultType.DEVICE_FAILURE,
            location="edge_0",
            tool_pred=lambda s: (
                s.action_type == DiagnosticAction.CHECK_STATUS
                and (s.parameters.get("device") == "edge_0" or s.parameters.get("source") == "edge_0")
            ),
            label="device_edge0_check_status",
        )
    )
    probes.append(
        _probe_one(
            model,
            protocol,
            fault=FaultType.PERFORMANCE_DEGRADATION,
            location="center->edge_0",
            tool_pred=lambda s: (
                s.action_type == DiagnosticAction.PING
                and s.parameters.get("source") == "center"
                and s.parameters.get("destination") == "edge_0"
            ),
            label="perf_center_edge0_ping",
        )
    )
    probes.append(
        _probe_one(
            model,
            protocol,
            fault=FaultType.MISCONFIGURATION,
            location="edge_0",
            tool_pred=lambda s: (
                s.action_type == DiagnosticAction.PING
                and s.parameters.get("source") == "edge_1"
                and s.parameters.get("destination") == "edge_0"
            ),
            label="misconfig_edge0_ping_to_gt",
        )
    )
    return probes


def _branch(cond: Dict, probes: List[Dict]) -> str:
    p_ec = cond["p_exact_cons_given_tool"]
    obs_moves = any(max(p["obs_max_abs_delta"].values()) > 0 for p in probes)
    phi_probes = [p for p in probes if p["phi_after_tool"] >= 1]
    policy_moves = any(p["tv_diagnosis"] > 0.02 for p in phi_probes) if phi_probes else False
    p_correct_up = any(
        p["p_correct_after_tool"] - p["p_correct_scan"] > 0.02 for p in phi_probes
    )
    if p_ec >= 0.05:
        return "branch_1_informed_exists_greedy_drops"
    if obs_moves and not policy_moves and p_ec < 0.02:
        return "branch_2_obs_changes_policy_ignores"
    if (not obs_moves) or any(
        p["label"].startswith("perf") and p["phi_after_tool"] >= 1 and p["obs_max_abs_delta"].get("discovery_matrix", 1) == 0
        and p["obs_max_abs_delta"].get("recent_diagnostics", 0) > 0
        for p in probes
    ):
        # finer: if phi rises but latency not in obs
        return "mixed_see_report"
    return "inconclusive"


def build_report(c: Dict, v: Dict, pc: List[Dict], pv: List[Dict]) -> str:
    def blk(name, d):
        return [
            f"### {name}",
            "",
            f"- n={d['n']}, tool-use={d['n_tool_use']} ({d['p_tool_use']:.0%}), Φ≥1={d['n_phi']} ({d['p_phi']:.0%})",
            f"- **P(Φ≥1 | tool-use)** = {d['p_phi_given_tool']:.1%}",
            f"- **P(exact ∧ consistent | tool-use)** = {d['p_exact_cons_given_tool']:.1%} ({d['n_exact_consistent']} exact-consistent total)",
            f"- **P(exact ∧ consistent | Φ≥1)** = {d['p_exact_cons_given_phi']:.1%}",
            "",
        ]

    def pblk(title, probes):
        lines = [f"### {title}", "", "| probe | Φ | max|Δobs| | TV(diagnose) | P(correct) scan→tool | feat0 changed | latency |", "|---|---:|---:|---:|---:|:---:|---:|"]
        for p in probes:
            md = max(p["obs_max_abs_delta"].values())
            lines.append(
                f"| {p['label']} | {p['phi_after_tool']} | {md:.4f} | {p['tv_diagnosis']:.3f} | "
                f"{p['p_correct_scan']:.3f}→{p['p_correct_after_tool']:.3f} | "
                f"{p['device_status_feat0_changed']} | {p['tool_latency_ms']} |"
            )
        lines.append("")
        return lines

    lines = [
        "# Stochastic evidence + observation conditioning audit",
        "",
        "> No retrain. E3 Control zip + Ablation-NL zip. NL-only stochastic eval.",
        f"> n={N_STOCH} episodes/model. Conditioning probes are controlled Star injects.",
        "",
        "## A — Tool-use conditionals (stochastic)",
        "",
        *blk("Control (E3 zip)", c),
        *blk("Ablation-NL", v),
        "## B — Does evidence reach obs / shift diagnose probs?",
        "",
        "Internal `DeviceStatus` is an enum (`UP`/`DOWN`). Vectorized `device_status[:,0]` is `1` iff `status == 'active'` (string). That match never fires, so feat0 stays 0.",
        "Latency lives in `connection_properties` and is **not** copied into `discovery_matrix` obs (int8 adjacency only). `recent_diagnostics` has success/hash/timestamp, not latency or error text.",
        "",
        *pblk("Control probes", pc),
        *pblk("Ablation-NL probes", pv),
        "## Branch",
        "",
    ]
    # interpretation filled after we see numbers in main
    return "\n".join(lines)


def main() -> None:
    protocol = load_protocol()
    ckpt = PROJECT_ROOT / "models" / "checkpoints"
    control = MaskablePPO.load(str((ckpt / "masked_ppo_e3_control_star.zip").with_suffix("")))
    variant = MaskablePPO.load(str((ckpt / "masked_ppo_ablation_nl_star.zip").with_suffix("")))

    print(f"Stochastic NL eval n={N_STOCH}...")
    raw_c = _evaluate_with_rows(
        control, protocol, fault_names=NONLINK, n_episodes=N_STOCH, seed_base=SEED_BASE, deterministic=False, agent="c_sto"
    )
    raw_v = _evaluate_with_rows(
        variant, protocol, fault_names=NONLINK, n_episodes=N_STOCH, seed_base=SEED_BASE, deterministic=False, agent="v_sto"
    )
    cond_c = _conditionals(raw_c)
    cond_v = _conditionals(raw_v)
    print("Control", cond_c)
    print("NL", cond_v)

    print("Conditioning probes...")
    probes_c = run_conditioning_probes(control, protocol)
    probes_v = run_conditioning_probes(variant, protocol)
    for p in probes_c + probes_v:
        print(p["label"], "phi", p["phi_after_tool"], "tv", round(p["tv_diagnosis"], 3), "dP", round(p["p_correct_after_tool"] - p["p_correct_scan"], 3), p["obs_max_abs_delta"])

    report = build_report(cond_c, cond_v, probes_c, probes_v)
    # Branch text from actual numbers
    def interpret(cond, probes, name):
        pec = cond["p_exact_cons_given_tool"]
        pphi = cond["p_phi_given_tool"]
        ev_probes = [p for p in probes if p["phi_after_tool"] >= 1]
        dummy = [p for p in probes if "healthy" in p["label"]]
        tv_ev = max((p["tv_diagnosis"] for p in ev_probes), default=0)
        tv_dummy = max((p["tv_diagnosis"] for p in dummy), default=0)
        dp = max((p["p_correct_after_tool"] - p["p_correct_scan"] for p in ev_probes), default=0)
        feat0 = any(p["device_status_feat0_changed"] for p in probes)
        perf = next((p for p in probes if p["label"].startswith("perf")), None)
        bits = []
        if pec < 0.02:
            bits.append(f"{name}: informed exact among tool-use is ~{pec:.1%} (branch-1 support is weak).")
        else:
            bits.append(f"{name}: informed exact among tool-use is {pec:.1%} — greedy is dropping a real informed path.")
        bits.append(f"P(Φ≥1|tool-use)={pphi:.1%} so most tools are not GT evidence.")
        if not feat0:
            bits.append("Vectorized device_status[:,0] never flips (status=='active' encoding miss).")
        if perf and perf["phi_after_tool"] >= 1 and perf["obs_max_abs_delta"].get("discovery_matrix", 1) == 0:
            bits.append("Perf: Φ rose from latency but discovery_matrix obs did not change — latency is not in the Gym vector.")
        if tv_ev < 0.02 and ev_probes:
            bits.append(f"After Φ-setting tools, diagnose TV={tv_ev:.3f} (policy almost ignores the new obs).")
        elif ev_probes:
            bits.append(f"After Φ-setting tools, diagnose TV={tv_ev:.3f}, ΔP(correct)={dp:+.3f} (dummy TV={tv_dummy:.3f}).")
        return bits

    extra = [
        "## Interpretation",
        "",
        *["- " + x for x in interpret(cond_c, probes_c, "Control")],
        *["- " + x for x in interpret(cond_v, probes_v, "Ablation-NL")],
        "",
        "E4 (diagnose-mask) only makes sense if evidence can condition the diagnose head.",
        "If obs omits latency / DOWN, or the policy TV stays ~0 after Φ tools, curriculum cannot create informed greedy.",
        "",
    ]
    text = report + "\n".join(extra)
    out = PROJECT_ROOT / "reports" / "stochastic_evidence_audit.md"
    out.write_text(text, encoding="utf-8")
    (PROJECT_ROOT / "results" / "stochastic_evidence_audit.json").write_text(
        json.dumps(
            {"control": cond_c, "ablation_nl": cond_v, "probes_control": probes_c, "probes_nl": probes_v},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
