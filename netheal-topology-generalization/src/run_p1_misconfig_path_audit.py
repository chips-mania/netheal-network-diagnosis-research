"""Misconfig path audit: 3-way failure split at Star N=8/10.

Buckets for non-success episodes:
  A. no_evidence — never reached GT-relevant Misconfig cues
  B. wrong_hypothesis — Rule `_try_conclude` fires but type/loc ≠ GT
  C. commit_fail — conclude_match reached but no successful diagnose
  D. incomplete_discrimination — partial GT cues, never conclude_match

Uses Generic (primary) + Baseline contrast. Policy does not see GT; labels are audit-only.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from netheal.environment.actions import ActionCategory
from netheal.faults.injector import FaultType

from .action_trace import _base_env, _format_target, _parse_action_spec
from .diagnosis_decomposition import location_match
from .env_utils import make_masked_eval_env, metrics_env
from .graph_obs import build_graph_obs
from .graph_policy import GraphPointerPolicy
from .pointer_actions import resolve_pointer
from .protocol import PROJECT_ROOT, load_protocol
from .rule_agent_v2 import RuleAgentV2
from .run_p1_commit_ablation import CKPT_GENERIC
from .run_p1_commit_fix import CKPT_BASE
from .run_p1_star10_failure_audit import evidence_flags
from .run_symmetric_reward_poc import set_seeds
from .star_pad import steps_budget


def misconfig_cues(mem, gt_loc: str) -> Dict[str, Any]:
    """Partial evidence toward Misconfig@gt_loc (audit-only)."""
    m = mem
    asym_from_gt = any(a == gt_loc and (b, a) not in m.missing for a, b in m.missing)
    asym_to_gt = any(b == gt_loc and (b, a) not in m.missing for a, b in m.missing)
    status_gt = m.status.get(gt_loc)
    iface = m.ifaces.get(gt_loc)
    iface_checked = iface is not None
    downs: List[str] = []
    if iface and iface.get("success"):
        downs = [
            str(i.get("destination"))
            for i in (iface.get("interfaces") or [])
            if str(i.get("status", "")).lower() == "down"
        ]
    return {
        "scanned": bool(m.scanned),
        "n_missing": len(m.missing),
        "asym_from_gt": asym_from_gt,
        "asym_to_gt": asym_to_gt,
        "status_gt": status_gt,
        "status_gt_up": status_gt == "up",
        "iface_checked_gt": iface_checked,
        "iface_downs_gt": downs,
        "partial": bool(
            asym_from_gt or asym_to_gt or (iface_checked and downs) or status_gt is not None
        ),
        "strong_partial": bool(
            (asym_from_gt and status_gt == "up")
            or (iface_checked and downs and asym_from_gt)
        ),
    }


def classify_episode(ep: Dict) -> str:
    if ep["success"]:
        return "success"
    if ep["conclude_match_any"]:
        return "C_commit_fail"
    if ep["wrong_conclude_any"]:
        return "B_wrong_hypothesis"
    if ep["strong_partial_any"] or ep["partial_any"]:
        return "D_incomplete_discrimination"
    return "A_no_evidence"


def load_policy(path, device) -> Optional[GraphPointerPolicy]:
    if not path.is_file():
        return None
    p = GraphPointerPolicy(hidden=64, layers=2).to(device)
    p.load_state_dict(torch.load(path, map_location=device))
    p.eval()
    return p


def audit_misconfig(
    policy: GraphPointerPolicy,
    protocol,
    *,
    n_devices: int,
    n: int,
    seed_base: int,
    device: torch.device,
) -> Dict:
    steps_max = steps_budget(n_devices, floor=int(protocol["max_episode_steps"]))
    episodes: List[Dict] = []

    for i in range(n):
        env = make_masked_eval_env(
            ["star"],
            fault_types=[FaultType.MISCONFIGURATION],
            protocol=protocol,
            evidence_obs=True,
            pbrs=True,
            pbrs_apply_shaping=False,
            min_devices=n_devices,
            max_devices=n_devices,
            max_episode_steps=steps_max,
        )
        agent_ev = RuleAgentV2()
        base = _base_env(env)
        obs, info = env.reset(seed=seed_base + i)
        gt = info.get("ground_truth_fault") or {}
        ft = gt.get("type", "")
        floc = gt.get("location", "")
        assert ft == FaultType.MISCONFIGURATION.value, ft
        agent_ev.reset()

        traj: List[str] = []
        conclude_match_any = False
        wrong_conclude_any = False
        partial_any = False
        strong_partial_any = False
        first_match_step = None
        first_wrong = None
        last_conclude: Optional[Tuple[str, str]] = None
        predicted_type = predicted_loc = None
        saw_dx = False
        step = 0
        max_cues: Dict[str, Any] = {}

        while True:
            step += 1
            cues = misconfig_cues(agent_ev.mem, floc)
            flags = evidence_flags(agent_ev.mem, ft, floc)
            if cues["partial"]:
                partial_any = True
            if cues["strong_partial"]:
                strong_partial_any = True
            max_cues = cues

            if flags.get("conclude_match"):
                conclude_match_any = True
                if first_match_step is None:
                    first_match_step = step
            elif flags.get("conclude_ready"):
                wrong_conclude_any = True
                agent = RuleAgentV2()
                agent.mem = agent_ev.mem
                conc = agent._try_conclude()
                if conc is not None:
                    last_conclude = conc
                    if first_wrong is None:
                        first_wrong = {"step": step, "type": conc[0], "loc": conc[1]}

            g = build_graph_obs(env, mem=agent_ev.mem)
            ptr = policy.act(g, device)
            traj.append(ptr.kind.value)
            action_id = int(resolve_pointer(env, ptr))
            obs, reward, terminated, truncated, info = env.step(action_id)
            agent_ev.observe_step(info)
            agent_ev.refresh_plan(env)

            spec = _parse_action_spec(info, base, action_id)
            phase = info.get("factorized_phase")
            if spec and spec.category == ActionCategory.DIAGNOSIS and phase != "type_commit":
                saw_dx = True
                predicted_type = (
                    spec.action_type.value if hasattr(spec.action_type, "value") else str(spec.action_type)
                )
                predicted_loc = _format_target(spec)

            if terminated or truncated:
                break

        # final flags after last tool
        cues = misconfig_cues(agent_ev.mem, floc)
        flags = evidence_flags(agent_ev.mem, ft, floc)
        if cues["partial"]:
            partial_any = True
        if cues["strong_partial"]:
            strong_partial_any = True
        if flags.get("conclude_match"):
            conclude_match_any = True
        elif flags.get("conclude_ready"):
            wrong_conclude_any = True
            agent = RuleAgentV2()
            agent.mem = agent_ev.mem
            conc = agent._try_conclude()
            if conc is not None:
                last_conclude = conc

        m = metrics_env(env).last_episode_metrics
        success = bool(m and m.diagnosis_success)
        timeout = (not success) and step >= steps_max
        ep = {
            "episode_id": i,
            "fault_location": floc,
            "success": success,
            "timeout": timeout,
            "submitted": saw_dx,
            "predicted_type": predicted_type,
            "predicted_location": predicted_loc,
            "steps": step,
            "traj_kinds": traj,
            "conclude_match_any": conclude_match_any,
            "wrong_conclude_any": wrong_conclude_any,
            "partial_any": partial_any,
            "strong_partial_any": strong_partial_any,
            "first_match_step": first_match_step,
            "first_wrong": first_wrong,
            "last_conclude": last_conclude,
            "final_cues": cues,
            "final_flags": {
                "conclude_match": flags.get("conclude_match"),
                "conclude_ready": flags.get("conclude_ready"),
                "conclude_type_ok": flags.get("conclude_type_ok"),
                "conclude_loc_ok": flags.get("conclude_loc_ok"),
            },
        }
        ep["bucket"] = classify_episode(ep)
        episodes.append(ep)
        env.close()

    counts = Counter(e["bucket"] for e in episodes)
    n_ep = len(episodes)
    # Wrong conclude type histogram
    wrong_types = Counter()
    for e in episodes:
        if e["bucket"] == "B_wrong_hypothesis" and e.get("last_conclude"):
            wrong_types[e["last_conclude"][0]] += 1

    samples = defaultdict(list)
    for e in episodes:
        if len(samples[e["bucket"]]) < 3:
            samples[e["bucket"]].append(
                {
                    "loc": e["fault_location"],
                    "timeout": e["timeout"],
                    "submitted": e["submitted"],
                    "pred": f"{e['predicted_type']}:{e['predicted_location']}",
                    "traj": " → ".join(e["traj_kinds"]),
                    "cues": {
                        "asym_from_gt": e["final_cues"].get("asym_from_gt"),
                        "status_gt": e["final_cues"].get("status_gt"),
                        "iface_checked_gt": e["final_cues"].get("iface_checked_gt"),
                        "iface_downs": e["final_cues"].get("iface_downs_gt"),
                        "n_missing": e["final_cues"].get("n_missing"),
                    },
                    "last_conclude": e.get("last_conclude"),
                    "first_match_step": e.get("first_match_step"),
                }
            )

    return {
        "n_devices": n_devices,
        "n": n_ep,
        "exact": sum(1 for e in episodes if e["success"]) / max(1, n_ep),
        "timeout": sum(1 for e in episodes if e["timeout"]) / max(1, n_ep),
        "bucket_counts": dict(counts),
        "bucket_rates": {k: counts[k] / max(1, n_ep) for k in counts},
        "wrong_conclude_types": dict(wrong_types),
        "cue_rates_all": {
            "partial_any": sum(1 for e in episodes if e["partial_any"]) / max(1, n_ep),
            "strong_partial_any": sum(1 for e in episodes if e["strong_partial_any"]) / max(1, n_ep),
            "conclude_match_any": sum(1 for e in episodes if e["conclude_match_any"]) / max(1, n_ep),
            "wrong_conclude_any": sum(1 for e in episodes if e["wrong_conclude_any"]) / max(1, n_ep),
            "iface_checked_gt": sum(1 for e in episodes if e["final_cues"].get("iface_checked_gt"))
            / max(1, n_ep),
            "asym_from_gt": sum(1 for e in episodes if e["final_cues"].get("asym_from_gt")) / max(1, n_ep),
            "status_gt_up": sum(1 for e in episodes if e["final_cues"].get("status_gt_up")) / max(1, n_ep),
        },
        "samples": dict(samples),
        "episodes": episodes,
    }


def _pct(x: float) -> str:
    return f"{x:.0%}"


def main() -> None:
    protocol = load_protocol()
    cfg = protocol.get("graph_expert_bc_poc", {})
    set_seeds(int(cfg.get("policy_seed", 1)))
    device = torch.device("cpu")
    n_eval = int(cfg.get("eval_episodes", 100))
    # Misconfig-only: use 50 eps for stabler rates (cycle is already single fault)
    n_mis = max(50, n_eval // 2)
    seed = int(cfg.get("eval_seed_base", 130000)) + 22000

    models = {
        "baseline": load_policy(CKPT_BASE, device),
        "generic": load_policy(CKPT_GENERIC, device),
    }
    results: Dict[str, Dict[int, Dict]] = {}
    for name, pol in models.items():
        if pol is None:
            print(f"skip {name}: missing ckpt")
            continue
        results[name] = {}
        for n_dev in (8, 10):
            print(f"=== Misconfig audit {name} N={n_dev} n={n_mis} ===")
            results[name][n_dev] = audit_misconfig(
                pol, protocol, n_devices=n_dev, n=n_mis, seed_base=seed + 1000 * n_dev, device=device
            )
            r = results[name][n_dev]
            print(
                f"  exact={r['exact']:.0%} buckets={r['bucket_rates']} "
                f"cues_match={r['cue_rates_all']['conclude_match_any']:.0%} "
                f"iface_gt={r['cue_rates_all']['iface_checked_gt']:.0%}"
            )

    # Narrative from Generic N=10
    g10 = results.get("generic", {}).get(10, {})
    rates = g10.get("bucket_rates") or {}
    dominant = max(rates.items(), key=lambda kv: kv[1])[0] if rates else "n/a"
    if dominant.startswith("D_") or dominant.startswith("A_"):
        narrative = (
            "Misconfig@N10 failures are dominated by **not reaching conclude-match** "
            f"({dominant}) — evidence-path / Link–Misconfig discrimination, not commit."
        )
    elif dominant.startswith("C_"):
        narrative = "Misconfig still shows residual **commit_fail** after Generic — unexpected; inspect samples."
    elif dominant.startswith("B_"):
        narrative = (
            "Misconfig failures often reach a **wrong Rule conclusion** (e.g. Link) — "
            "hypothesis discrimination error on shared missing-edge cues."
        )
    else:
        narrative = "See bucket table."

    lines = [
        "# Misconfig path audit (N=8 / N=10)",
        "",
        "> Fault-only eval. Buckets: A no_evidence · B wrong_hypothesis · C commit_fail · D incomplete_discrimination · success",
        f"> Primary model: **Generic**. {narrative}",
        "",
        "## Bucket rates",
        "",
        "| Model | N | Exact | A no_ev | B wrong | C commit | D incomplete |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        "A_no_evidence",
        "B_wrong_hypothesis",
        "C_commit_fail",
        "D_incomplete_discrimination",
        "success",
    ]
    for name in ("baseline", "generic"):
        if name not in results:
            continue
        for n_dev in (8, 10):
            r = results[name][n_dev]
            br = r["bucket_rates"]
            lines.append(
                f"| {name} | {n_dev} | {_pct(r['exact'])} | {_pct(br.get('A_no_evidence', 0))} | "
                f"{_pct(br.get('B_wrong_hypothesis', 0))} | {_pct(br.get('C_commit_fail', 0))} | "
                f"{_pct(br.get('D_incomplete_discrimination', 0))} |"
            )

    lines.extend(
        [
            "",
            "## Cue rates (end of episode)",
            "",
            "| Model | N | partial | strong_partial | conclude_match | wrong_conclude | iface@GT | asym_from_GT |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("baseline", "generic"):
        if name not in results:
            continue
        for n_dev in (8, 10):
            c = results[name][n_dev]["cue_rates_all"]
            lines.append(
                f"| {name} | {n_dev} | {_pct(c['partial_any'])} | {_pct(c['strong_partial_any'])} | "
                f"{_pct(c['conclude_match_any'])} | {_pct(c['wrong_conclude_any'])} | "
                f"{_pct(c['iface_checked_gt'])} | {_pct(c['asym_from_gt'])} |"
            )

    if g10.get("wrong_conclude_types"):
        lines.extend(["", "## Generic N=10 wrong conclude types", ""])
        for t, c in sorted(g10["wrong_conclude_types"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {t}: {c}")

    lines.extend(["", "## Samples (Generic N=10)", ""])
    for bucket, items in (g10.get("samples") or {}).items():
        lines.append(f"### {bucket}")
        for s in items:
            lines.append(
                f"- loc=`{s['loc']}` timeout={s['timeout']} submitted={s['submitted']} pred=`{s['pred']}`"
            )
            lines.append(f"  - traj: `{s['traj']}`")
            lines.append(f"  - cues: {s['cues']} last_conclude={s['last_conclude']}")

    lines.extend(
        [
            "",
            "## Readout",
            "",
            narrative,
            "",
            "Next: if D/A dominate → tool-order / iface targeting for Misconfig discrimination; "
            "if B → Link vs Misconfig feature confusion; if C → residual commit (unlikely after Generic).",
            "",
            "Run: `python -m src.run_p1_misconfig_path_audit`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results_dir = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (reports / "p1_misconfig_path_audit.md").write_text("\n".join(lines), encoding="utf-8")

    slim = {}
    for name, by_n in results.items():
        slim[name] = {}
        for n_dev, block in by_n.items():
            slim[name][str(n_dev)] = {k: v for k, v in block.items() if k != "episodes"}

    (results_dir / "p1_misconfig_path_audit.json").write_text(
        json.dumps({"narrative": narrative, "results": slim}, indent=2, default=str),
        encoding="utf-8",
    )
    print(narrative)
    print("Wrote", reports / "p1_misconfig_path_audit.md")


if __name__ == "__main__":
    main()
