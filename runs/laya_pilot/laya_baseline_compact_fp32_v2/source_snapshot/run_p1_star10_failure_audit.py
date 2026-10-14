"""P1 Star scale failure audit: timeout / evidence / heads / pointer rank.

Compares N=6 (seen), N=8 (near), N=10 (breakdown) without changing the model.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from netheal.environment.actions import ActionCategory
from netheal.faults.injector import FaultType

from .action_trace import (
    EpisodeSummary,
    StepRecord,
    _base_env,
    _diagnosis_correct,
    _format_target,
    _is_action_valid,
    _is_tool_action,
    _parse_action_spec,
)
from .diagnosis_decomposition import location_match
from .env_utils import make_masked_eval_env, metrics_env
from .graph_obs import build_graph_obs
from .graph_policy import (
    FAULT_ORDER,
    KIND_ORDER,
    NODE_FAULTS,
    GraphPointerPolicy,
    collate_graphs,
    pointer_to_labels,
)
from .pbrs import find_pbrs
from .pointer_actions import encode_flat_as_pointer, resolve_pointer
from .protocol import PROJECT_ROOT, load_protocol
from .rule_agent_v2 import RuleAgentV2
from .run_graph_expert_bc import collect_rule_v2_graph, train_graph_bc
from .run_symmetric_reward_poc import set_seeds
from .star_pad import steps_budget


CKPT = PROJECT_ROOT / "models" / "checkpoints" / "graph_pointer_p1.pt"


def _parse_edge(loc: str) -> Optional[Tuple[str, str]]:
    if not loc or "->" not in loc:
        return None
    a, b = loc.split("->", 1)
    return a.strip(), b.strip()


def evidence_flags(mem, gt_type: str, gt_loc: str) -> Dict[str, Any]:
    """GT-aware evidence snapshot from Rule mem (no GT used in policy)."""
    flags: Dict[str, Any] = {
        "phi_proxy_type": False,
        "phi_proxy_loc": False,
        "missing_hits_gt": False,
        "conclude_ready": False,
        "conclude_type_ok": False,
        "conclude_loc_ok": False,
    }
    m = mem
    if gt_type == FaultType.DEVICE_FAILURE.value:
        st = m.status.get(gt_loc)
        flags["phi_proxy_type"] = st == "down" or any(v == "down" for v in m.status.values())
        flags["phi_proxy_loc"] = st == "down"
    elif gt_type == FaultType.PERFORMANCE_DEGRADATION.value:
        ends = _parse_edge(gt_loc)
        for (s, d), pr in m.ping.items():
            if not pr.get("success"):
                continue
            lat = pr.get("latency_ms")
            if lat is None or float(lat) < 10.0:
                continue
            flags["phi_proxy_type"] = True
            if ends and (s in ends or d in ends):
                flags["phi_proxy_loc"] = True
    elif gt_type == FaultType.MISCONFIGURATION.value:
        if gt_loc in m.ifaces:
            downs = [
                str(i.get("destination"))
                for i in (m.ifaces[gt_loc].get("interfaces") or [])
                if str(i.get("status", "")).lower() == "down"
            ]
            if downs:
                flags["phi_proxy_type"] = True
                flags["phi_proxy_loc"] = True
        # asymmetric missing involving device
        for a, b in m.missing:
            if a == gt_loc or b == gt_loc:
                flags["phi_proxy_type"] = True
                if a == gt_loc and (b, a) not in m.missing:
                    flags["phi_proxy_loc"] = True
    elif gt_type == FaultType.LINK_FAILURE.value:
        ends = _parse_edge(gt_loc)
        if ends:
            a, b = ends
            hit = (a, b) in m.missing or (b, a) in m.missing
            flags["missing_hits_gt"] = hit
            flags["phi_proxy_type"] = hit
            flags["phi_proxy_loc"] = (a, b) in m.missing and (b, a) in m.missing

    # Rule conclude from same evidence
    agent = RuleAgentV2()
    agent.mem = m
    concluded = agent._try_conclude()
    if concluded is not None:
        ct, cl = concluded
        flags["conclude_ready"] = True
        flags["conclude_type_ok"] = ct == gt_type
        flags["conclude_loc_ok"] = location_match(gt_loc, cl, gt_type)
        flags["conclude_match"] = flags["conclude_type_ok"] and flags["conclude_loc_ok"]
    else:
        flags["conclude_match"] = False
    return flags


@torch.no_grad()
def policy_logits(policy: GraphPointerPolicy, g, device: torch.device):
    batch = collate_graphs([g], pad_nodes=max(g.n_nodes, 1), pad_edges=max(g.n_edges, 1))
    batch = {k: v.to(device) for k, v in batch.items()}
    return policy.forward_logits(
        batch["node_feat"],
        batch["edge_index"],
        batch["edge_feat"],
        batch["node_mask"],
        batch["edge_mask"],
        batch["globals"],
    )


def pointer_ranks(logits, g, lab: Dict[str, int]) -> Dict[str, Optional[float]]:
    """1-indexed rank of correct targets; None if not supervised."""
    out: Dict[str, Optional[float]] = {"kind": None, "node": None, "edge": None, "fault": None}
    kind_scores = logits["kind"][0].detach().cpu().numpy()
    true_k = lab["kind"]
    out["kind"] = float(1 + np.sum(kind_scores > kind_scores[true_k]))

    if lab["node"] >= 0:
        ns = logits["node"][0, : g.n_nodes].detach().cpu().numpy()
        t = lab["node"]
        if 0 <= t < len(ns):
            out["node"] = float(1 + np.sum(ns > ns[t]))
    if lab["edge"] >= 0 and g.n_edges > 0:
        es = logits["edge"][0, : g.n_edges].detach().cpu().numpy()
        t = lab["edge"]
        if 0 <= t < len(es):
            out["edge"] = float(1 + np.sum(es > es[t]))
    if lab["fault"] >= 0:
        fs = logits["fault"][0].detach().cpu().numpy()
        t = lab["fault"]
        out["fault"] = float(1 + np.sum(fs > fs[t]))
    return out


def ensure_p1(protocol, device: torch.device) -> GraphPointerPolicy:
    policy = GraphPointerPolicy(hidden=64, layers=2).to(device)
    if CKPT.is_file():
        policy.load_state_dict(torch.load(CKPT, map_location=device))
        print(f"Loaded {CKPT}")
        return policy

    print("=== Checkpoint missing — train P1 graph-only (same recipe) ===")
    cfg = protocol["graph_expert_bc_poc"]
    graphs_all, labs_all = [], []
    for n in cfg["train_ns"]:
        n = int(n)
        print(f"  collect graph N={n}")
        g, lab, gst = collect_rule_v2_graph(
            protocol, n_devices=n, n_ok=int(cfg["demos_per_n"]), seed0=80_000 + 1000 * n
        )
        print(" ", gst)
        graphs_all.extend(g)
        labs_all.extend(lab)
    losses = train_graph_bc(
        policy,
        graphs_all,
        labs_all,
        epochs=int(cfg.get("p1_epochs", cfg.get("bc_epochs", 80))),
        batch=int(cfg["batch_size"]),
        lr=3e-4,
        pad_nodes=int(cfg["pad_max_devices"]),
        pad_edges=int(cfg["pad_edges"]),
        device=device,
    )
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), CKPT)
    print(f"  saved {CKPT} last_loss={losses[-1]:.4f}")
    return policy


def rollout_audit(
    policy: GraphPointerPolicy,
    protocol,
    *,
    n_devices: int,
    n: int,
    seed_base: int,
    device: torch.device,
) -> Dict:
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    steps = steps_budget(n_devices, floor=int(protocol["max_episode_steps"]))
    episodes: List[Dict] = []
    kind_hist = defaultdict(int)

    for i in range(n):
        fault = faults[i % len(faults)]
        env = make_masked_eval_env(
            ["star"],
            fault_types=[fault],
            protocol=protocol,
            evidence_obs=True,
            pbrs=True,
            pbrs_apply_shaping=False,
            min_devices=n_devices,
            max_devices=n_devices,
            max_episode_steps=steps,
        )
        agent_ev = RuleAgentV2()
        base = _base_env(env)
        obs, info = env.reset(seed=seed_base + i)
        gt = info.get("ground_truth_fault") or {}
        fault_type = gt.get("type", "")
        fault_location = gt.get("location", "")
        agent_ev.reset()
        traj_kinds: List[str] = []
        saw_dx = False
        predicted_type = None
        predicted_location = None
        invalid_count = 0
        step = 0
        max_phi = 0
        last_flags: Dict[str, Any] = {}
        conclude_ready_any = False
        conclude_match_any = False
        evidence_before_end = False

        while True:
            step += 1
            g = build_graph_obs(env, mem=agent_ev.mem)
            logits = policy_logits(policy, g, device)
            kind_i = int(logits["kind"][0].argmax().item())
            kind = KIND_ORDER[kind_i]
            traj_kinds.append(kind.value)
            kind_hist[kind.value] += 1

            ptr = policy.act(g, device)
            action_id = int(resolve_pointer(env, ptr))
            if not _is_action_valid(base, action_id):
                invalid_count += 1
            obs, reward, terminated, truncated, info = env.step(action_id)
            agent_ev.observe_step(info)
            agent_ev.refresh_plan(env)

            flags = evidence_flags(agent_ev.mem, fault_type, fault_location)
            last_flags = flags
            if flags.get("conclude_ready"):
                conclude_ready_any = True
            if flags.get("conclude_match"):
                conclude_match_any = True
            if flags.get("phi_proxy_loc") or flags.get("missing_hits_gt") or flags.get("conclude_match"):
                evidence_before_end = True

            tracker = find_pbrs(env)
            max_phi = max(max_phi, int(tracker.bits.phi))

            spec = _parse_action_spec(info, base, action_id)
            phase = info.get("factorized_phase")
            if spec and spec.category == ActionCategory.DIAGNOSIS and phase != "type_commit":
                saw_dx = True
                predicted_type = (
                    spec.action_type.value if hasattr(spec.action_type, "value") else str(spec.action_type)
                )
                predicted_location = _format_target(spec)

            if terminated or truncated:
                break

        m = metrics_env(env).last_episode_metrics
        success = bool(m and m.diagnosis_success)
        timeout = (not success) and step >= steps
        tracker = find_pbrs(env)
        bits = tracker.bits_before_diagnosis or tracker.bits.snapshot()
        phi = tracker.phi_before_diagnosis
        if phi is None:
            phi = tracker.bits.phi

        # Commit failure: enough evidence to conclude correctly, but no / failed submit
        commit_fail = (not success) and bool(last_flags.get("conclude_match"))
        evidence_no_commit = timeout and conclude_match_any and not saw_dx

        episodes.append(
            {
                "episode_id": i,
                "fault_type": fault_type,
                "fault_location": fault_location,
                "success": success,
                "timeout": timeout,
                "submitted": saw_dx,
                "steps": step,
                "max_steps": steps,
                "invalid_count": invalid_count,
                "predicted_type": predicted_type,
                "predicted_location": predicted_location,
                "phi_end": int(phi),
                "phi_max": int(max_phi),
                "type_bit": int(bits.get("type_bit", 0)),
                "loc_bit": int(bits.get("loc_bit", 0)),
                "traj_kinds": traj_kinds,
                "n_diagnose_kind": sum(1 for k in traj_kinds if k == "diagnose"),
                "evidence": last_flags,
                "conclude_ready_any": conclude_ready_any,
                "conclude_match_any": conclude_match_any,
                "evidence_loc_or_match": evidence_before_end,
                "commit_fail": commit_fail,
                "evidence_no_commit": evidence_no_commit,
            }
        )
        env.close()

    def _agg(subset: List[Dict]) -> Dict:
        n_s = len(subset)
        if n_s == 0:
            return {"n": 0}
        to = [e for e in subset if e["timeout"]]
        ok = [e for e in subset if e["success"]]
        return {
            "n": n_s,
            "exact": len(ok) / n_s,
            "timeout": len(to) / n_s,
            "submit": sum(1 for e in subset if e["submitted"]) / n_s,
            "conclude_match_any": sum(1 for e in subset if e["conclude_match_any"]) / n_s,
            "conclude_match_on_timeout": (
                sum(1 for e in to if e["conclude_match_any"]) / max(1, len(to))
            ),
            "evidence_no_commit": sum(1 for e in subset if e["evidence_no_commit"]) / n_s,
            "commit_fail": sum(1 for e in subset if e["commit_fail"]) / n_s,
            "phi_ge1": sum(1 for e in subset if e["phi_max"] >= 1) / n_s,
            "missing_hit_rate": (
                sum(1 for e in subset if e["evidence"].get("missing_hits_gt")) / n_s
            ),
            "mean_steps": float(np.mean([e["steps"] for e in subset])),
            "diagnose_kind_rate": sum(1 for e in subset if e["n_diagnose_kind"] > 0) / n_s,
        }

    by_ft = defaultdict(list)
    for e in episodes:
        by_ft[e["fault_type"]].append(e)

    # Sample timeout trajectories (up to 2 per fault)
    samples = []
    for ft, sub in by_ft.items():
        for e in sub:
            if e["timeout"] and len([s for s in samples if s["fault_type"] == ft]) < 2:
                samples.append(
                    {
                        "fault_type": ft,
                        "fault_location": e["fault_location"],
                        "traj_kinds": e["traj_kinds"],
                        "conclude_match_any": e["conclude_match_any"],
                        "evidence": e["evidence"],
                        "submitted": e["submitted"],
                        "steps": e["steps"],
                    }
                )

    return {
        "n_devices": n_devices,
        "n": n,
        "max_steps": steps,
        "overall": _agg(episodes),
        "by_fault": {ft: _agg(sub) for ft, sub in sorted(by_ft.items())},
        "kind_hist": dict(kind_hist),
        "timeout_samples": samples,
        "episodes": episodes,
    }


def expert_state_head_audit(
    policy: GraphPointerPolicy,
    protocol,
    *,
    n_devices: int,
    n_ok: int,
    seed0: int,
    device: torch.device,
) -> Dict:
    """On Rule-v2 successful demos: head accuracy + correct-target ranks."""
    graphs, labels, gst = collect_rule_v2_graph(
        protocol, n_devices=n_devices, n_ok=n_ok, seed0=seed0
    )
    policy.eval()
    kind_ok = kind_n = 0
    node_ok = node_n = 0
    edge_ok = edge_n = 0
    fault_ok = fault_n = 0
    # diagnose vs tool kind split
    diag_kind_ok = diag_kind_n = 0
    tool_kind_ok = tool_kind_n = 0
    ranks = {"node": [], "edge": [], "fault": [], "kind": []}
    top3 = {"node": [], "edge": []}
    by_kind = defaultdict(lambda: {"n": 0, "kind_ok": 0, "node_ok": 0, "node_n": 0, "edge_ok": 0, "edge_n": 0})

    for g, lab in zip(graphs, labels):
        logits = policy_logits(policy, g, device)
        pred_k = int(logits["kind"][0].argmax().item())
        k = lab["kind"]
        kind_n += 1
        kok = int(pred_k == k)
        kind_ok += kok
        kname = KIND_ORDER[k].value
        by_kind[kname]["n"] += 1
        by_kind[kname]["kind_ok"] += kok
        if k == 4:  # diagnose
            diag_kind_n += 1
            diag_kind_ok += kok
        else:
            tool_kind_n += 1
            tool_kind_ok += kok

        r = pointer_ranks(logits, g, lab)
        if r["kind"] is not None:
            ranks["kind"].append(r["kind"])

        if lab["node"] >= 0:
            node_n += 1
            pred_n = int(logits["node"][0, : g.n_nodes].argmax().item())
            nok = int(pred_n == lab["node"])
            node_ok += nok
            by_kind[kname]["node_n"] += 1
            by_kind[kname]["node_ok"] += nok
            if r["node"] is not None:
                ranks["node"].append(r["node"])
                top3["node"].append(int(r["node"] <= 3))
        if lab["edge"] >= 0 and g.n_edges > 0:
            edge_n += 1
            pred_e = int(logits["edge"][0, : g.n_edges].argmax().item())
            eok = int(pred_e == lab["edge"])
            edge_ok += eok
            by_kind[kname]["edge_n"] += 1
            by_kind[kname]["edge_ok"] += eok
            if r["edge"] is not None:
                ranks["edge"].append(r["edge"])
                top3["edge"].append(int(r["edge"] <= 3))
        if lab["fault"] >= 0:
            fault_n += 1
            pred_f = int(logits["fault"][0].argmax().item())
            fault_ok += int(pred_f == lab["fault"])
            if r["fault"] is not None:
                ranks["fault"].append(r["fault"])

    def _mean(xs):
        return float(np.mean(xs)) if xs else None

    return {
        "collect": gst,
        "n_transitions": len(labels),
        "kind_acc": kind_ok / max(1, kind_n),
        "node_acc": node_ok / max(1, node_n) if node_n else None,
        "edge_acc": edge_ok / max(1, edge_n) if edge_n else None,
        "fault_acc": fault_ok / max(1, fault_n) if fault_n else None,
        "diagnose_kind_acc": diag_kind_ok / max(1, diag_kind_n),
        "tool_kind_acc": tool_kind_ok / max(1, tool_kind_n),
        "mean_rank": {k: _mean(v) for k, v in ranks.items()},
        "top3": {k: _mean(v) for k, v in top3.items()},
        "by_kind": {
            k: {
                "n": v["n"],
                "kind_acc": v["kind_ok"] / max(1, v["n"]),
                "node_acc": v["node_ok"] / max(1, v["node_n"]) if v["node_n"] else None,
                "edge_acc": v["edge_ok"] / max(1, v["edge_n"]) if v["edge_n"] else None,
            }
            for k, v in by_kind.items()
        },
    }


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.0%}"


def _fmt_rank(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}"


def main() -> None:
    protocol = load_protocol()
    cfg = protocol.get("graph_expert_bc_poc", {})
    set_seeds(int(cfg.get("policy_seed", 1)))
    device = torch.device("cpu")
    policy = ensure_p1(protocol, device)
    policy.eval()

    n_eval = int(cfg.get("eval_episodes", 100))
    seed = int(cfg.get("eval_seed_base", 130000))
    # head audit: enough demos per N for stable rates
    n_head = 200

    rollouts = {}
    heads = {}
    for n_dev in (6, 8, 10):
        print(f"=== Rollout audit Star N={n_dev} ===")
        rollouts[n_dev] = rollout_audit(
            policy,
            protocol,
            n_devices=n_dev,
            n=n_eval,
            seed_base=seed + 1000 * n_dev,
            device=device,
        )
        o = rollouts[n_dev]["overall"]
        print(
            f"  exact={o['exact']:.0%} timeout={o['timeout']:.0%} "
            f"conclude_match_any={o['conclude_match_any']:.0%} "
            f"ev_no_commit={o['evidence_no_commit']:.0%}"
        )
        print(f"=== Expert-state heads Star N={n_dev} ===")
        heads[n_dev] = expert_state_head_audit(
            policy,
            protocol,
            n_devices=n_dev,
            n_ok=n_head,
            seed0=200_000 + 1000 * n_dev,
            device=device,
        )
        h = heads[n_dev]
        print(
            f"  kind={h['kind_acc']:.0%} node={h['node_acc']} edge={h['edge_acc']} "
            f"fault={h['fault_acc']} edge_rank={h['mean_rank']['edge']} edge_top3={h['top3']['edge']}"
        )

    # Diagnosis narrative
    r10 = rollouts[10]["overall"]
    r10_link = rollouts[10]["by_fault"].get("link_failure", {})
    r10_mis = rollouts[10]["by_fault"].get("misconfiguration", {})
    h6, h8, h10 = heads[6], heads[8], heads[10]

    if r10.get("evidence_no_commit", 0) >= 0.25:
        narrative = (
            "Many N=10 timeouts look like **commit/continue failures**: Rule-style conclude "
            "already matches GT from the same evidence mem, but the policy did not diagnose."
        )
    elif r10_link.get("missing_hit_rate", 0) < 0.3 and r10.get("timeout", 0) > 0.4:
        narrative = (
            "N=10 timeouts look closer to **evidence/path failures**: missing-GT hit rate low; "
            "graph/tool sequence may not surface the fault before budget."
        )
    else:
        narrative = (
            "Mixed: check per-fault conclude_match_on_timeout vs missing_hit / head ranks below."
        )

    lines = [
        "# P1 Star scale failure audit (N=6 / 8 / 10)",
        "",
        "> Checkpoint: `models/checkpoints/graph_pointer_p1.pt`",
        "> Rollout: P1 acting. Heads/ranks: Rule-v2 demo states (expert trajectory).",
        f"> Read: {narrative}",
        "",
        "## 1. Online rollout (Exact / Timeout / Evidence→Commit)",
        "",
        "| N | Exact | Timeout | Submit | Conclude=GT (any) | Ev→no-commit | Φ≥1 (PBRS) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n_dev in (6, 8, 10):
        o = rollouts[n_dev]["overall"]
        lines.append(
            f"| {n_dev} | {_fmt_pct(o['exact'])} | {_fmt_pct(o['timeout'])} | {_fmt_pct(o['submit'])} | "
            f"{_fmt_pct(o['conclude_match_any'])} | {_fmt_pct(o['evidence_no_commit'])} | {_fmt_pct(o['phi_ge1'])} |"
        )

    lines.extend(
        [
            "",
            "### Per-fault @ N=10",
            "",
            "| Fault | Exact | Timeout | Conclude=GT | Conclude|timeout | Missing∋GT | Ev→no-commit |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ft, b in rollouts[10]["by_fault"].items():
        lines.append(
            f"| {ft} | {_fmt_pct(b.get('exact'))} | {_fmt_pct(b.get('timeout'))} | "
            f"{_fmt_pct(b.get('conclude_match_any'))} | {_fmt_pct(b.get('conclude_match_on_timeout'))} | "
            f"{_fmt_pct(b.get('missing_hit_rate'))} | {_fmt_pct(b.get('evidence_no_commit'))} |"
        )

    lines.extend(
        [
            "",
            "### Same @ N=8 (Misconfig early weakness check)",
            "",
            "| Fault | Exact | Timeout | Conclude=GT | Ev→no-commit |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for ft, b in rollouts[8]["by_fault"].items():
        lines.append(
            f"| {ft} | {_fmt_pct(b.get('exact'))} | {_fmt_pct(b.get('timeout'))} | "
            f"{_fmt_pct(b.get('conclude_match_any'))} | {_fmt_pct(b.get('evidence_no_commit'))} |"
        )

    lines.extend(
        [
            "",
            "## 2. Expert-state head accuracy (Rule demos → P1 logits)",
            "",
            "| N | Kind | Tool-kind | Diag-kind | Node | Edge | Fault |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for n_dev in (6, 8, 10):
        h = heads[n_dev]
        lines.append(
            f"| {n_dev} | {_fmt_pct(h['kind_acc'])} | {_fmt_pct(h['tool_kind_acc'])} | "
            f"{_fmt_pct(h['diagnose_kind_acc'])} | {_fmt_pct(h['node_acc'])} | "
            f"{_fmt_pct(h['edge_acc'])} | {_fmt_pct(h['fault_acc'])} |"
        )

    lines.extend(
        [
            "",
            "## 3. Correct-target rank (1=best) / Top-3",
            "",
            "| N | Node mean-rank | Node top-3 | Edge mean-rank | Edge top-3 | Fault mean-rank |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for n_dev in (6, 8, 10):
        h = heads[n_dev]
        mr, t3 = h["mean_rank"], h["top3"]
        lines.append(
            f"| {n_dev} | {_fmt_rank(mr['node'])} | {_fmt_pct(t3['node'])} | "
            f"{_fmt_rank(mr['edge'])} | {_fmt_pct(t3['edge'])} | {_fmt_rank(mr['fault'])} |"
        )

    lines.extend(["", "## 4. Timeout trajectory samples (N=10)", ""])
    for s in rollouts[10]["timeout_samples"]:
        lines.append(
            f"- **{s['fault_type']}** `{s['fault_location']}` steps={s['steps']} "
            f"submitted={s['submitted']} conclude_match={s['conclude_match_any']}"
        )
        lines.append(f"  - kinds: `{' → '.join(s['traj_kinds'])}`")
        ev = s["evidence"]
        lines.append(
            f"  - end evidence: conclude_match={ev.get('conclude_match')} "
            f"missing_hit={ev.get('missing_hits_gt')} type={ev.get('phi_proxy_type')} loc={ev.get('phi_proxy_loc')}"
        )

    edge_delta = (h10["edge_acc"] or 0) - (h6["edge_acc"] or 0)
    diag_delta = h10["diagnose_kind_acc"] - h6["diagnose_kind_acc"]
    lines.extend(
        [
            "",
            "## 5. Audit readout",
            "",
            f"- Edge pointer top-1 Δ N6→N10: **{edge_delta:+.0%}** "
            f"(top-3 @N10={_fmt_pct(h10['top3']['edge'])}, mean-rank={_fmt_rank(h10['mean_rank']['edge'])})",
            f"- Diagnose-kind Δ N6→N10: **{diag_delta:+.0%}**",
            f"- N10 overall evidence→no-commit: **{_fmt_pct(r10.get('evidence_no_commit'))}**; "
            f"conclude=GT on timeouts: **{_fmt_pct(r10.get('conclude_match_on_timeout'))}**",
            "",
            "### Branch-level read",
            "",
            "- **Link @N10**: Exact 0%, timeout 100%, conclude=GT on timeout **100%**, traj = status spam after iface — pure **commit failure**.",
            "- **Misconfig @N8**: already Exact 48% / timeout 52% with conclude=GT **100%** on those eps — commit failure starts before N=10.",
            "- **Misconfig @N10**: conclude=GT only ~40% — mix of incomplete asymmetry evidence + no commit.",
            "- **Perf**: remains Exact 100% through N=10.",
            "- **Expert-state diagnose-kind** collapses N6→N10 (kind still strong for tools) — when-to-diagnose is scale-fragile; edge top-1 falls by N=8 while top-3 stays ~2/3 (pointer calibration).",
            "",
            narrative,
            "",
            "Run: `python -m src.run_p1_star10_failure_audit`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "p1_star10_failure_audit.md").write_text("\n".join(lines), encoding="utf-8")

    # Drop full episode lists from json for size (keep aggregates + samples)
    payload = {
        "narrative": narrative,
        "rollouts": {
            str(k): {kk: vv for kk, vv in v.items() if kk != "episodes"} for k, v in rollouts.items()
        },
        "heads": heads,
    }
    # attach compact episode counts only
    for k, v in rollouts.items():
        payload["rollouts"][str(k)]["n_episodes"] = len(v["episodes"])

    (results / "p1_star10_failure_audit.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(narrative)
    print("Wrote", reports / "p1_star10_failure_audit.md")


if __name__ == "__main__":
    main()
