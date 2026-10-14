"""Misconfig cue→node pointer rank audit (N=6/8/10).

After asymmetric-missing involving GT appears, measure node-head rank of GT
(and of Rule's preferred status/iface target). Also offline Rule-demo node acc.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from netheal.faults.injector import FaultType

from .env_utils import make_masked_eval_env
from .graph_obs import build_graph_obs
from .graph_policy import GraphPointerPolicy
from .pointer_actions import encode_flat_as_pointer, resolve_pointer
from .protocol import PROJECT_ROOT, load_protocol
from .rule_agent_v2 import RuleAgentV2
from .run_p1_commit_ablation import CKPT_GENERIC
from .run_p1_commit_fix import CKPT_BASE
from .run_p1_misconfig_path_audit import misconfig_cues
from .run_p1_star10_failure_audit import policy_logits
from .run_symmetric_reward_poc import set_seeds
from .star_pad import steps_budget


def node_rank_stats(logits, g, target_id: str) -> Optional[Dict[str, float]]:
    if target_id not in g.node_index:
        return None
    ti = g.node_index[target_id]
    scores = logits["node"][0, : g.n_nodes].detach().cpu().numpy()
    probs = F.softmax(torch.tensor(scores), dim=-1).numpy()
    rank = float(1 + np.sum(scores > scores[ti]))
    top_i = int(np.argmax(scores))
    return {
        "rank": rank,
        "top1": float(top_i == ti),
        "top3": float(rank <= 3),
        "p_gt": float(probs[ti]),
        "p_top": float(probs[top_i]),
        "top_id": g.node_ids[top_i],
        "margin": float(scores[ti] - max([scores[j] for j in range(len(scores)) if j != ti], default=scores[ti])),
    }


def rule_preferred_node(mem, gt_loc: str) -> Optional[str]:
    """Heuristic: Rule would status missing endpoints then iface sources — prefer GT if queued."""
    # Prefer GT if it is an asymmetric missing source
    for a, b in sorted(mem.missing):
        if a == gt_loc and (b, a) not in mem.missing:
            return gt_loc
    for a, b in sorted(mem.missing):
        if b == gt_loc or a == gt_loc:
            return a if a == gt_loc else b
    return gt_loc


def load_policy(path, device):
    if not path.is_file():
        return None
    p = GraphPointerPolicy(hidden=64, layers=2).to(device)
    p.load_state_dict(torch.load(path, map_location=device))
    p.eval()
    return p


def forced_cue_rank(
    policy: GraphPointerPolicy,
    protocol,
    *,
    n_devices: int,
    n: int,
    seed_base: int,
    device: torch.device,
) -> Dict:
    """Reach asym-missing cue via forced scan (not policy), then score node-head on GT.

    Isolates representation binding from kind-head rollout failure.
    """
    from .scripted_expert import find_scan

    steps_max = steps_budget(n_devices, floor=int(protocol["max_episode_steps"]))
    first_cue_rows: List[Dict] = []
    spoke_rows: List[Dict] = []
    center_rows: List[Dict] = []

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
        agent = RuleAgentV2()
        obs, info = env.reset(seed=seed_base + i)
        gt = info.get("ground_truth_fault") or {}
        floc = gt.get("location", "")
        agent.reset()
        # forced scan
        from .action_mask import base_env as _be

        root = _be(env)
        scan_id = find_scan(root)
        obs, reward, term, trunc, info = env.step(int(scan_id))
        agent.observe_step(info)
        agent.refresh_plan(env)
        cues = misconfig_cues(agent.mem, floc)
        g = build_graph_obs(env, mem=agent.mem)
        logits = policy_logits(policy, g, device)
        if cues.get("asym_from_gt") or cues.get("asym_to_gt") or cues.get("partial"):
            st = node_rank_stats(logits, g, floc)
            if st is not None:
                row = {
                    "fault_location": floc,
                    "is_center": floc == "center",
                    "gt": st,
                }
                first_cue_rows.append(row)
                (center_rows if floc == "center" else spoke_rows).append(row)
        env.close()

    def agg(rows: List[Dict]) -> Dict[str, Any]:
        stats = [r["gt"] for r in rows if r.get("gt")]
        if not stats:
            return {"n": 0}
        return {
            "n": len(stats),
            "mean_rank": float(np.mean([s["rank"] for s in stats])),
            "top1": float(np.mean([s["top1"] for s in stats])),
            "top3": float(np.mean([s["top3"] for s in stats])),
            "p_gt_mean": float(np.mean([s["p_gt"] for s in stats])),
            "margin_mean": float(np.mean([s["margin"] for s in stats])),
        }

    return {
        "n_devices": n_devices,
        "n_episodes": n,
        "first_cue_gt": agg(first_cue_rows),
        "first_cue_spoke": agg(spoke_rows),
        "first_cue_center": agg(center_rows),
        "when_status_chosen": {"n": 0},
        "when_iface_chosen": {"n": 0},
    }


def online_cue_rank(
    policy: GraphPointerPolicy,
    protocol,
    *,
    n_devices: int,
    n: int,
    seed_base: int,
    device: torch.device,
) -> Dict:
    steps_max = steps_budget(n_devices, floor=int(protocol["max_episode_steps"]))
    first_cue_rows: List[Dict] = []
    status_choice_rows: List[Dict] = []
    iface_choice_rows: List[Dict] = []
    spoke_rows: List[Dict] = []
    center_rows: List[Dict] = []

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
        agent = RuleAgentV2()
        obs, info = env.reset(seed=seed_base + i)
        gt = info.get("ground_truth_fault") or {}
        floc = gt.get("location", "")
        agent.reset()
        saw_cue = False
        step = 0

        while True:
            step += 1
            cues = misconfig_cues(agent.mem, floc)
            g = build_graph_obs(env, mem=agent.mem)
            logits = policy_logits(policy, g, device)

            if cues.get("asym_from_gt") and not saw_cue:
                saw_cue = True
                st = node_rank_stats(logits, g, floc)
                if st is not None:
                    pref = rule_preferred_node(agent.mem, floc)
                    pref_st = node_rank_stats(logits, g, pref) if pref else None
                    row = {
                        "fault_location": floc,
                        "is_center": floc == "center",
                        "gt": st,
                        "pref": pref_st,
                        "pref_id": pref,
                    }
                    first_cue_rows.append(row)
                    (center_rows if floc == "center" else spoke_rows).append(row)

            # If policy picks status/iface, record whether GT is top
            kind_i = int(logits["kind"][0].argmax().item())
            ptr = policy.act(g, device)
            if ptr.kind.value == "check_status":
                st = node_rank_stats(logits, g, floc)
                if st is not None:
                    status_choice_rows.append(
                        {"gt_rank": st["rank"], "gt_top1": st["top1"], "chosen": ptr.node, "gt": floc}
                    )
            elif ptr.kind.value == "check_interfaces":
                st = node_rank_stats(logits, g, floc)
                if st is not None:
                    iface_choice_rows.append(
                        {
                            "gt_rank": st["rank"],
                            "gt_top1": st["top1"],
                            "chosen": ptr.node,
                            "gt": floc,
                            "chose_gt": float(ptr.node == floc),
                        }
                    )

            action_id = int(resolve_pointer(env, ptr))
            obs, reward, terminated, truncated, info = env.step(action_id)
            agent.observe_step(info)
            agent.refresh_plan(env)
            if terminated or truncated:
                break
        env.close()

    def agg(rows: List[Dict], key_path: str = "gt") -> Dict[str, Any]:
        if not rows:
            return {"n": 0}
        if key_path == "gt":
            stats = [r["gt"] for r in rows if r.get("gt")]
        else:
            stats = rows
        if not stats:
            return {"n": 0}
        if key_path == "gt":
            return {
                "n": len(stats),
                "mean_rank": float(np.mean([s["rank"] for s in stats])),
                "top1": float(np.mean([s["top1"] for s in stats])),
                "top3": float(np.mean([s["top3"] for s in stats])),
                "p_gt_mean": float(np.mean([s["p_gt"] for s in stats])),
                "margin_mean": float(np.mean([s["margin"] for s in stats])),
            }
        return {"n": len(stats)}

    iface_chose_gt = (
        float(np.mean([r["chose_gt"] for r in iface_choice_rows])) if iface_choice_rows else None
    )

    return {
        "n_devices": n_devices,
        "n_episodes": n,
        "first_cue_gt": agg(first_cue_rows, "gt"),
        "first_cue_spoke": agg(spoke_rows, "gt"),
        "first_cue_center": agg(center_rows, "gt"),
        "when_status_chosen": {
            "n": len(status_choice_rows),
            "gt_top1": float(np.mean([r["gt_top1"] for r in status_choice_rows]))
            if status_choice_rows
            else None,
            "gt_mean_rank": float(np.mean([r["gt_rank"] for r in status_choice_rows]))
            if status_choice_rows
            else None,
        },
        "when_iface_chosen": {
            "n": len(iface_choice_rows),
            "chose_gt": iface_chose_gt,
            "gt_top1": float(np.mean([r["gt_top1"] for r in iface_choice_rows]))
            if iface_choice_rows
            else None,
            "gt_mean_rank": float(np.mean([r["gt_rank"] for r in iface_choice_rows]))
            if iface_choice_rows
            else None,
        },
    }


def rule_misconfig_pointer_eval(
    policy: GraphPointerPolicy,
    protocol,
    *,
    n_devices: int,
    n_ok: int,
    seed0: int,
    device: torch.device,
) -> Dict:
    """Rule-v2 successful Misconfig demos: node rank at expert status/iface steps."""
    from .action_mask import base_env
    from .action_trace import _diagnosis_correct, _parse_action_spec

    max_steps = steps_budget(n_devices, floor=int(protocol["max_episode_steps"]))
    ranks_status, ranks_iface, ranks_first_cue = [], [], []
    chose = {"status_gt": 0, "status_n": 0, "iface_gt": 0, "iface_n": 0}
    ok = tries = 0
    while ok < n_ok and tries < n_ok * 10:
        tries += 1
        env = make_masked_eval_env(
            ["star"],
            fault_types=[FaultType.MISCONFIGURATION],
            protocol=protocol,
            evidence_obs=True,
            pbrs=False,
            min_devices=n_devices,
            max_devices=n_devices,
            max_episode_steps=max_steps,
        )
        agent = RuleAgentV2()
        obs, info = env.reset(seed=seed0 + tries)
        gt = info.get("ground_truth_fault") or {}
        floc = gt.get("location", "")
        agent.reset()
        success = False
        saw_cue = False
        root = base_env(env)
        for t in range(1, max_steps + 1):
            cues = misconfig_cues(agent.mem, floc)
            g = build_graph_obs(env, mem=agent.mem)
            try:
                action = int(agent.choose(env, obs, t, []))
            except RuntimeError:
                break
            ptr = encode_flat_as_pointer(env, action)
            logits = policy_logits(policy, g, device)

            if cues.get("asym_from_gt") and not saw_cue:
                saw_cue = True
                st = node_rank_stats(logits, g, floc)
                if st:
                    ranks_first_cue.append(st)

            if ptr is not None and ptr.kind.value == "check_status":
                st = node_rank_stats(logits, g, floc)
                if st:
                    ranks_status.append(st)
                    pred = int(logits["node"][0, : g.n_nodes].argmax().item())
                    chose["status_n"] += 1
                    chose["status_gt"] += int(g.node_ids[pred] == floc)
            if ptr is not None and ptr.kind.value == "check_interfaces":
                st = node_rank_stats(logits, g, floc)
                if st:
                    ranks_iface.append(st)
                    pred = int(logits["node"][0, : g.n_nodes].argmax().item())
                    chose["iface_n"] += 1
                    chose["iface_gt"] += int(g.node_ids[pred] == floc)

            obs, reward, term, trunc, info = env.step(action)
            agent.observe_step(info)
            agent.refresh_plan(env)
            spec = _parse_action_spec(info, root, action)
            if spec is not None and _diagnosis_correct(info, spec) is True:
                success = True
            if term or trunc:
                break
        env.close()
        if success:
            ok += 1

    def summ(xs):
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean_rank": float(np.mean([x["rank"] for x in xs])),
            "top1": float(np.mean([x["top1"] for x in xs])),
            "top3": float(np.mean([x["top3"] for x in xs])),
            "p_gt_mean": float(np.mean([x["p_gt"] for x in xs])),
            "margin_mean": float(np.mean([x["margin"] for x in xs])),
        }

    return {
        "ok_episodes": ok,
        "tries": tries,
        "first_cue_gt": summ(ranks_first_cue),
        "at_expert_status": summ(ranks_status),
        "at_expert_iface": summ(ranks_iface),
        "pred_is_gt_when_status": chose["status_gt"] / max(1, chose["status_n"]),
        "pred_is_gt_when_iface": chose["iface_gt"] / max(1, chose["iface_n"]),
        "n_status": chose["status_n"],
        "n_iface": chose["iface_n"],
    }


def _pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:.0%}"


def _f(x, nd=2) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def main() -> None:
    protocol = load_protocol()
    cfg = protocol.get("graph_expert_bc_poc", {})
    set_seeds(int(cfg.get("policy_seed", 1)))
    device = torch.device("cpu")
    n_ep = 50
    seed = int(cfg.get("eval_seed_base", 130000)) + 33000

    models = {
        "baseline": load_policy(CKPT_BASE, device),
        "generic": load_policy(CKPT_GENERIC, device),
    }
    online = {}
    offline = {}
    for name, pol in models.items():
        if pol is None:
            print(f"skip {name}")
            continue
        online[name] = {}
        offline[name] = {}
        for n_dev in (6, 8, 10):
            print(f"=== Online cue-rank {name} N={n_dev} ===")
            online[name][n_dev] = online_cue_rank(
                pol, protocol, n_devices=n_dev, n=n_ep, seed_base=seed + 100 * n_dev, device=device
            )
            r = online[name][n_dev]["first_cue_gt"]
            print(
                f"  first_cue GT top1={_pct(r.get('top1'))} top3={_pct(r.get('top3'))} "
                f"rank={_f(r.get('mean_rank'))} spoke_top1={_pct(online[name][n_dev]['first_cue_spoke'].get('top1'))}"
            )
            print(f"=== Offline Rule-state {name} N={n_dev} ===")
            offline[name][n_dev] = rule_misconfig_pointer_eval(
                pol,
                protocol,
                n_devices=n_dev,
                n_ok=40,
                seed0=400_000 + 1000 * n_dev + (0 if name == "generic" else 50),
                device=device,
            )
            o = offline[name][n_dev]["first_cue_gt"]
            print(
                f"  rule-cue GT top1={_pct(o.get('top1'))} rank={_f(o.get('mean_rank'))} "
                f"iface_pred_gt={_pct(offline[name][n_dev].get('pred_is_gt_when_iface'))}"
            )

    g10 = online.get("generic", {}).get(10, {})
    fc = g10.get("first_cue_gt") or {}
    sp = g10.get("first_cue_spoke") or {}
    if (fc.get("top3") or 0) >= 0.7 and (fc.get("top1") or 0) < 0.5:
        narrative = (
            "After asym-missing cue @N10, GT node is often in **top-3 but not top-1** "
            "→ pointer **calibration/ranking**, not total binding failure."
        )
    elif (fc.get("top1") or 0) < 0.2 and (sp.get("top1") or 0) < 0.15:
        narrative = (
            "After asym-missing cue @N10, GT spoke rank collapses (top-1 very low) "
            "→ **cue→node binding** fails to generalize under scale."
        )
    else:
        narrative = "See tables — split center vs spoke carefully."

    lines = [
        "# Misconfig cue→node pointer rank audit",
        "",
        "> After `asym_from_GT` first appears: node-head rank of GT. Online = policy rollout; Offline = Rule trajectory states.",
        f"> {narrative}",
        "",
        "## Online: first asym-missing cue — GT node rank",
        "",
        "| Model | N | n | top-1 | top-3 | mean-rank | P(GT) | margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "generic"):
        if name not in online:
            continue
        for n_dev in (6, 8, 10):
            r = online[name][n_dev]["first_cue_gt"]
            lines.append(
                f"| {name} | {n_dev} | {r.get('n', 0)} | {_pct(r.get('top1'))} | {_pct(r.get('top3'))} | "
                f"{_f(r.get('mean_rank'))} | {_f(r.get('p_gt_mean'), 3)} | {_f(r.get('margin_mean'), 3)} |"
            )

    lines.extend(
        [
            "",
            "## Online: first cue — center vs spoke (Generic)",
            "",
            "| N | Center top-1 | Spoke top-1 | Spoke top-3 | Spoke mean-rank |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for n_dev in (6, 8, 10):
        if "generic" not in online:
            break
        c = online["generic"][n_dev]["first_cue_center"]
        s = online["generic"][n_dev]["first_cue_spoke"]
        lines.append(
            f"| {n_dev} | {_pct(c.get('top1'))} | {_pct(s.get('top1'))} | {_pct(s.get('top3'))} | {_f(s.get('mean_rank'))} |"
        )

    lines.extend(
        [
            "",
            "## Online: when policy chooses iface — picks GT?",
            "",
            "| Model | N | n_iface | chose GT | GT top-1 in scores |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("baseline", "generic"):
        if name not in online:
            continue
        for n_dev in (6, 8, 10):
            w = online[name][n_dev]["when_iface_chosen"]
            lines.append(
                f"| {name} | {n_dev} | {w.get('n', 0)} | {_pct(w.get('chose_gt'))} | {_pct(w.get('gt_top1'))} |"
            )

    lines.extend(
        [
            "",
            "## Offline Rule demos: first cue + expert iface steps (GT rank)",
            "",
            "| Model | N | cue top-1 | cue rank | iface steps pred=GT | iface GT top-1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("baseline", "generic"):
        if name not in offline:
            continue
        for n_dev in (6, 8, 10):
            o = offline[name][n_dev]
            fc = o.get("first_cue_gt") or {}
            iface = o.get("at_expert_iface") or {}
            lines.append(
                f"| {name} | {n_dev} | {_pct(fc.get('top1'))} | {_f(fc.get('mean_rank'))} | "
                f"{_pct(o.get('pred_is_gt_when_iface'))} | {_pct(iface.get('top1'))} |"
            )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            narrative,
            "",
            "If spoke top-3 high / top-1 low → ranking curriculum. If top-3 also low → binding/representation.",
            "Next: pointer-only supervised probe (kind fixed, target only).",
            "",
            "Run: `python -m src.run_p1_misconfig_pointer_rank`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "p1_misconfig_pointer_rank.md").write_text("\n".join(lines), encoding="utf-8")
    (results / "p1_misconfig_pointer_rank.json").write_text(
        json.dumps({"narrative": narrative, "online": online, "offline": offline}, indent=2, default=str),
        encoding="utf-8",
    )
    print(narrative)
    print("Wrote", reports / "p1_misconfig_pointer_rank.md")


if __name__ == "__main__":
    main()
