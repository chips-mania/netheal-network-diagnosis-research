"""Pointer curriculum: oversample post-scan cue → next diagnostic node (Star 4/5/6 only).

Does not add N=8/10 to train. Does not change graph features.
Eval: forced-scan first-cue GT node rank (primary) + optional Misconfig Exact.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from netheal.faults.injector import FaultType

from .action_mask import base_env
from .action_trace import _diagnosis_correct, _parse_action_spec
from .env_utils import make_masked_eval_env
from .graph_obs import build_graph_obs
from .graph_policy import KIND_ORDER, GraphPointerPolicy, bc_loss, collate_graphs, pointer_to_labels
from .pointer_actions import encode_flat_as_pointer
from .protocol import PROJECT_ROOT, load_protocol
from .rule_agent_v2 import RuleAgentV2
from .run_graph_expert_bc import collect_rule_v2_graph, evaluate_graph_policy
from .run_p1_commit_ablation import CKPT_GENERIC
from .run_p1_commit_fix import CKPT_BASE
from .run_p1_misconfig_pointer_rank import forced_cue_rank, load_policy
from .run_p1_pointer_only_probe import CKPT_PTR, filter_node_steps, train_node_only
from .run_star_scale_poc import _pf
from .run_symmetric_reward_poc import set_seeds
from .scripted_expert import find_scan
from .star_pad import steps_budget

CKPT_CURR = PROJECT_ROOT / "models" / "checkpoints" / "graph_pointer_cue_curriculum.pt"
STATUS_I = next(i for i, k in enumerate(KIND_ORDER) if k.value == "check_status")
IFACE_I = next(i for i, k in enumerate(KIND_ORDER) if k.value == "check_interfaces")


def collect_post_scan_targets(
    protocol,
    *,
    n_devices: int,
    n_ok: int,
    seed0: int,
) -> Tuple[List, List[Dict], Dict]:
    """After scan+plan: graph → Rule's next status/iface node (endpoint of missing)."""
    max_steps = steps_budget(n_devices, floor=int(protocol["max_episode_steps"]))
    graphs, labels = [], []
    ok = tries = 0
    per = {}
    while ok < n_ok and tries < n_ok * 10:
        tries += 1
        # Mix faults so missing-edge cues appear (link/misconfig) and also status paths
        faults = [FaultType(ft) for ft in protocol["fault_types"]]
        fault = faults[tries % len(faults)]
        env = make_masked_eval_env(
            ["star"],
            fault_types=[fault],
            protocol=protocol,
            evidence_obs=True,
            pbrs=False,
            min_devices=n_devices,
            max_devices=n_devices,
            max_episode_steps=max_steps,
        )
        agent = RuleAgentV2()
        obs, info = env.reset(seed=seed0 + tries)
        gt = (info.get("ground_truth_fault") or {}).get("type", "unknown")
        agent.reset()
        root = base_env(env)
        # forced scan
        obs, reward, term, trunc, info = env.step(int(find_scan(root)))
        agent.observe_step(info)
        agent.refresh_plan(env)
        if not agent.mem.missing and not agent.mem.status_queue and not agent.mem.iface_queue:
            env.close()
            continue
        # Rule's next investigation node after plan
        target = None
        if agent.mem.status_queue:
            target = agent.mem.status_queue[0]
        elif agent.mem.iface_queue:
            target = agent.mem.iface_queue[0]
        g = build_graph_obs(env, mem=agent.mem)
        if target is None or target not in g.node_index:
            env.close()
            continue
        ni = g.node_index[target]
        # Prefer status kind if status_queue else iface
        kind = STATUS_I if agent.mem.status_queue else IFACE_I
        graphs.append(g)
        labels.append({"kind": kind, "node": ni, "edge": -1, "fault": -1})
        # Also label another endpoint of a missing edge (binding both ends)
        if agent.mem.missing:
            a, b = sorted(agent.mem.missing)[0]
            for ep in (a, b):
                if ep in g.node_index and ep != target:
                    graphs.append(g)
                    labels.append(
                        {
                            "kind": STATUS_I,
                            "node": g.node_index[ep],
                            "edge": -1,
                            "fault": -1,
                        }
                    )
                    break
        ok += 1
        per[gt] = per.get(gt, 0) + 1
        env.close()
    return graphs, labels, {"ok": ok, "tries": tries, "per_fault_ok": per, "n_transitions": len(labels)}


def oversample(graphs, labels, *, copies: int):
    if copies <= 1:
        return graphs, labels
    return graphs + graphs * (copies - 1), labels + labels * (copies - 1)


def train_curriculum(
    policy: GraphPointerPolicy,
    base_g,
    base_l,
    cue_g,
    cue_l,
    *,
    epochs: int,
    batch: int,
    lr: float,
    pad_nodes: int,
    pad_edges: int,
    device: torch.device,
    w_cue: float,
) -> List[float]:
    """Mix normal node BC + weighted post-scan cue batches; keep light kind CE on base."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    losses = []
    policy.train()
    n_b, n_c = len(base_l), len(cue_l)
    for ep in range(epochs):
        ep_loss = []
        # base node/full mini-epochs
        idx_b = np.random.permutation(n_b)
        idx_c = np.random.permutation(n_c)
        # equal number of steps roughly
        n_steps = max(n_b, n_c)
        for start in range(0, n_steps, batch):
            loss = torch.tensor(0.0, device=device)
            # base batch
            sel_b = idx_b[start % n_b : start % n_b + batch]
            if len(sel_b) < 2:
                sel_b = idx_b[: max(2, min(batch, n_b))]
            gs = [base_g[i] for i in sel_b]
            bt = {k: v.to(device) for k, v in collate_graphs(gs, pad_nodes=pad_nodes, pad_edges=pad_edges).items()}
            lab = {
                "kind": torch.tensor([base_l[i]["kind"] for i in sel_b], device=device),
                "node": torch.tensor([base_l[i]["node"] for i in sel_b], device=device),
                "edge": torch.tensor([base_l[i]["edge"] for i in sel_b], device=device),
                "fault": torch.tensor([base_l[i]["fault"] for i in sel_b], device=device),
            }
            logits = policy.forward_logits(
                bt["node_feat"], bt["edge_index"], bt["edge_feat"], bt["node_mask"], bt["edge_mask"], bt["globals"]
            )
            loss = loss + bc_loss(logits, lab)
            # cue batch
            sel_c = idx_c[start % n_c : start % n_c + batch]
            if len(sel_c) < 2:
                sel_c = idx_c[: max(2, min(batch, n_c))]
            gs_c = [cue_g[i] for i in sel_c]
            bt_c = {
                k: v.to(device)
                for k, v in collate_graphs(gs_c, pad_nodes=pad_nodes, pad_edges=pad_edges).items()
            }
            y = torch.tensor([cue_l[i]["node"] for i in sel_c], device=device, dtype=torch.long)
            logits_c = policy.forward_logits(
                bt_c["node_feat"],
                bt_c["edge_index"],
                bt_c["edge_feat"],
                bt_c["node_mask"],
                bt_c["edge_mask"],
                bt_c["globals"],
            )
            loss = loss + w_cue * F.cross_entropy(logits_c["node"], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss.append(float(loss.item()))
        losses.append(float(np.mean(ep_loss)) if ep_loss else 0.0)
        if (ep + 1) % max(1, epochs // 4) == 0 or ep + 1 == epochs:
            print(f"  cue-curr epoch {ep+1}/{epochs} loss={losses[-1]:.4f}")
    return losses


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
    cfg = protocol["graph_expert_bc_poc"]
    set_seeds(int(cfg["policy_seed"]))
    device = torch.device("cpu")
    pad = int(cfg["pad_max_devices"])
    demos = int(cfg["demos_per_n"])
    cue_ok = int(cfg.get("cue_curriculum_demos_per_n", 400))
    cue_copies = int(cfg.get("cue_curriculum_copies", 4))
    w_cue = float(cfg.get("cue_curriculum_w", 3.0))
    epochs = int(cfg.get("cue_curriculum_epochs", 50))
    n_rank = 50
    seed = int(cfg["eval_seed_base"]) + 55000

    # Base node steps from full Rule BC collect
    graphs_all, labs_all = [], []
    for n in cfg["train_ns"]:
        n = int(n)
        print(f"=== Collect Rule graph N={n} ===")
        g, lab, gst = collect_rule_v2_graph(
            protocol, n_devices=n, n_ok=demos, seed0=80_000 + 1000 * n
        )
        print(" ", gst)
        graphs_all.extend(g)
        labs_all.extend(lab)
    base_g, base_l = filter_node_steps(graphs_all, labs_all)

    # Post-scan curriculum
    cue_g, cue_l = [], []
    cue_stats = {}
    for n in cfg["train_ns"]:
        n = int(n)
        print(f"=== Collect post-scan cue targets N={n} ===")
        g, lab, st = collect_post_scan_targets(
            protocol, n_devices=n, n_ok=cue_ok, seed0=600_000 + 1000 * n
        )
        print(" ", st)
        cue_stats[n] = st
        cue_g.extend(g)
        cue_l.extend(lab)
    cue_g, cue_l = oversample(cue_g, cue_l, copies=cue_copies)
    print(f"=== Train: base_node={len(base_l)} cue={len(cue_l)} w_cue={w_cue} ===")

    policy = GraphPointerPolicy(hidden=64, layers=2).to(device)
    if CKPT_BASE.is_file():
        policy.load_state_dict(torch.load(CKPT_BASE, map_location=device))
        print(f"Warm-start {CKPT_BASE}")

    losses = train_curriculum(
        policy,
        base_g,
        base_l,
        cue_g,
        cue_l,
        epochs=epochs,
        batch=int(cfg["batch_size"]),
        lr=3e-4,
        pad_nodes=pad,
        pad_edges=int(cfg["pad_edges"]),
        device=device,
        w_cue=w_cue,
    )
    CKPT_CURR.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), CKPT_CURR)
    policy.eval()

    # Compare ranks
    refs = {
        "baseline_joint": load_policy(CKPT_BASE, device),
        "generic_joint": load_policy(CKPT_GENERIC, device),
        "node_only": load_policy(CKPT_PTR, device),
        "cue_curriculum": policy,
    }
    ranks = {}
    for name, pol in refs.items():
        if pol is None:
            continue
        ranks[name] = {}
        for n_dev in (6, 8, 10):
            print(f"=== Forced-cue {name} N={n_dev} ===")
            ranks[name][n_dev] = forced_cue_rank(
                pol, protocol, n_devices=n_dev, n=n_rank, seed_base=seed + 100 * n_dev, device=device
            )
            s = ranks[name][n_dev]["first_cue_spoke"]
            print(
                f"  spoke top1={_pct(s.get('top1'))} top3={_pct(s.get('top3'))} rank={_f(s.get('mean_rank'))}"
            )

    # Misconfig exact for curriculum vs baseline/generic (full policy still has kind)
    n_eval = int(cfg.get("eval_episodes", 100))
    eval_seed = int(cfg["eval_seed_base"])
    mis_exact = {}
    for name, pol in (("baseline", refs.get("baseline_joint")), ("curriculum", policy)):
        if pol is None:
            continue
        mis_exact[name] = {}
        for n_dev in (8, 10):
            # evaluate_graph_policy is mixed faults — run misconfig-only via forced env loop? use evaluate with note
            # Quick: 50 misconfig episodes via evaluate is heavy; use rank-primary and short exact
            from .run_p1_misconfig_path_audit import audit_misconfig

            print(f"=== Misconfig exact {name} N={n_dev} ===")
            block = audit_misconfig(
                pol, protocol, n_devices=n_dev, n=50, seed_base=eval_seed + 8000 + n_dev, device=device
            )
            mis_exact[name][n_dev] = {
                "exact": block["exact"],
                "bucket_rates": block["bucket_rates"],
                "cue_rates": block["cue_rates_all"],
            }
            print(f"  exact={block['exact']:.0%} buckets={block['bucket_rates']}")

    c10 = ranks.get("cue_curriculum", {}).get(10, {}).get("first_cue_spoke") or {}
    n10 = ranks.get("node_only", {}).get(10, {}).get("first_cue_spoke") or {}
    g10 = ranks.get("generic_joint", {}).get(10, {}).get("first_cue_spoke") or {}
    t1 = c10.get("top1") or 0
    t3 = c10.get("top3") or 0
    prev = n10.get("top1") or 0
    if t1 >= 0.55:
        decision = "curriculum_helps_binding"
    elif t1 >= prev + 0.10:
        decision = "curriculum_partial_gain"
    else:
        decision = "curriculum_insufficient_try_features"

    lines = [
        "# Pointer curriculum (post-scan cue → target node)",
        "",
        "> Train N∈{4,5,6} only. Oversample post-scan missing→endpoint node labels. No feature change. No N=8/10 in train.",
        f"> Decision: **{decision}**",
        "",
        f"- cue demos/n≈{cue_ok}, copies×{cue_copies}, w_cue={w_cue}, epochs={epochs}",
        f"- ckpt: `{CKPT_CURR.as_posix()}` last_loss={losses[-1]:.4f}",
        "",
        "## First-cue spoke GT rank (forced scan)",
        "",
        "| Model | N | top-1 | top-3 | mean-rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("baseline_joint", "generic_joint", "node_only", "cue_curriculum"):
        if name not in ranks:
            continue
        for n_dev in (6, 8, 10):
            r = ranks[name][n_dev]["first_cue_spoke"]
            lines.append(
                f"| {name} | {n_dev} | {_pct(r.get('top1'))} | {_pct(r.get('top3'))} | {_f(r.get('mean_rank'))} |"
            )

    lines.extend(
        [
            "",
            "## First-cue all GT",
            "",
            "| Model | N | top-1 | top-3 | mean-rank |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("baseline_joint", "node_only", "cue_curriculum"):
        if name not in ranks:
            continue
        for n_dev in (6, 8, 10):
            r = ranks[name][n_dev]["first_cue_gt"]
            lines.append(
                f"| {name} | {n_dev} | {_pct(r.get('top1'))} | {_pct(r.get('top3'))} | {_f(r.get('mean_rank'))} |"
            )

    lines.extend(["", "## Misconfig Exact (50 eps)", ""])
    lines.append("| Model | N | Exact | D incomplete | C commit | iface@GT |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in ("baseline", "curriculum"):
        if name not in mis_exact:
            continue
        for n_dev in (8, 10):
            b = mis_exact[name][n_dev]
            br = b["bucket_rates"]
            cr = b["cue_rates"]
            lines.append(
                f"| {name} | {n_dev} | {_pct(b['exact'])} | {_pct(br.get('D_incomplete_discrimination', 0))} | "
                f"{_pct(br.get('C_commit_fail', 0))} | {_pct(cr.get('iface_checked_gt'))} |"
            )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Spoke@N10 top-1: Generic {_pct(g10.get('top1'))} → node_only {_pct(prev)} → curriculum {_pct(t1)} (top-3 {_pct(t3)})",
            f"- Decision: **{decision}**",
            "",
            "If curriculum_insufficient → next: observation-derived missing-incident node features (still no GT leak).",
            "",
            "Run: `python -m src.run_p1_pointer_curriculum`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "p1_pointer_curriculum.md").write_text("\n".join(lines), encoding="utf-8")
    (results / "p1_pointer_curriculum.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "losses": losses,
                "cue_stats": cue_stats,
                "ranks": ranks,
                "mis_exact": mis_exact,
                "recipe": {
                    "cue_ok": cue_ok,
                    "copies": cue_copies,
                    "w_cue": w_cue,
                    "epochs": epochs,
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(decision)
    print("Wrote", reports / "p1_pointer_curriculum.md")


if __name__ == "__main__":
    main()
