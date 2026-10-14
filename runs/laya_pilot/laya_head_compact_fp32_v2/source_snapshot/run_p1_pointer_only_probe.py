"""Pointer-only probe: supervise node target (status/iface/node-dx), ignore kind/edge.

Train on Rule-v2 Star{4,5,6} node-labeled steps. Eval cue→GT rank on Misconfig N=6/8/10
vs frozen Generic joint policy — isolates whether binding is learnable without multi-head.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .graph_policy import KIND_ORDER, GraphPointerPolicy, collate_graphs
from .protocol import PROJECT_ROOT, load_protocol
from .run_graph_expert_bc import collect_rule_v2_graph
from .run_p1_commit_ablation import CKPT_GENERIC
from .run_p1_commit_fix import CKPT_BASE
from .run_p1_misconfig_pointer_rank import (
    forced_cue_rank,
    load_policy,
    rule_misconfig_pointer_eval,
)
from .run_symmetric_reward_poc import set_seeds

# kinds that supervise node pointer
NODE_KINDS = {
    next(i for i, k in enumerate(KIND_ORDER) if k.value == "check_status"),
    next(i for i, k in enumerate(KIND_ORDER) if k.value == "check_interfaces"),
    next(i for i, k in enumerate(KIND_ORDER) if k.value == "diagnose"),
}

CKPT_PTR = PROJECT_ROOT / "models" / "checkpoints" / "graph_pointer_node_only.pt"


def filter_node_steps(graphs, labels) -> Tuple[List, List]:
    g2, l2 = [], []
    for g, lab in zip(graphs, labels):
        if lab["kind"] in NODE_KINDS and lab["node"] >= 0:
            g2.append(g)
            l2.append(lab)
    return g2, l2


def train_node_only(
    policy: GraphPointerPolicy,
    graphs: List,
    labels: List[Dict],
    *,
    epochs: int,
    batch: int,
    lr: float,
    pad_nodes: int,
    pad_edges: int,
    device: torch.device,
    freeze_kind: bool = True,
) -> List[float]:
    if freeze_kind:
        for p in policy.kind_head.parameters():
            p.requires_grad = False
        for p in policy.fault_head.parameters():
            p.requires_grad = False
        for p in policy.edge_score.parameters():
            p.requires_grad = False
    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=lr)
    n = len(labels)
    losses = []
    policy.train()
    # keep frozen heads in eval mode behavior? still forward them but no grad
    for ep in range(epochs):
        idx = np.random.permutation(n)
        ep_loss = []
        for start in range(0, n, batch):
            sel = idx[start : start + batch]
            if len(sel) < 2:
                continue
            gs = [graphs[i] for i in sel]
            bt = collate_graphs(gs, pad_nodes=pad_nodes, pad_edges=pad_edges)
            bt = {k: v.to(device) for k, v in bt.items()}
            y = torch.tensor([labels[i]["node"] for i in sel], device=device, dtype=torch.long)
            logits = policy.forward_logits(
                bt["node_feat"], bt["edge_index"], bt["edge_feat"], bt["node_mask"], bt["edge_mask"], bt["globals"]
            )
            # mask invalid pad nodes already -1e9 in forward
            loss = F.cross_entropy(logits["node"], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss.append(float(loss.item()))
        losses.append(float(np.mean(ep_loss)) if ep_loss else 0.0)
        if (ep + 1) % max(1, epochs // 4) == 0 or ep + 1 == epochs:
            print(f"  node-only epoch {ep+1}/{epochs} loss={losses[-1]:.4f}")
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
    epochs = int(cfg.get("pointer_probe_epochs", 60))
    n_ep = 50
    seed = int(cfg["eval_seed_base"]) + 44000

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

    graphs_n, labs_n = filter_node_steps(graphs_all, labs_all)
    print(f"=== Node-only data: {len(labs_n)}/{len(labs_all)} transitions ===")

    # Train node-only WITHOUT destroying kind via encoder-only updates:
    # warm-start full model, train node + encoder, keep small kind CE so encoder stays usable.
    # Primary eval uses forced_cue_rank (scan forced) so kind is irrelevant for the claim.
    policy = GraphPointerPolicy(hidden=64, layers=2).to(device)
    if CKPT_BASE.is_file():
        policy.load_state_dict(torch.load(CKPT_BASE, map_location=device))
        print(f"Warm-start from {CKPT_BASE}")

    losses = train_node_only(
        policy,
        graphs_n,
        labs_n,
        epochs=epochs,
        batch=int(cfg["batch_size"]),
        lr=3e-4,
        pad_nodes=pad,
        pad_edges=int(cfg["pad_edges"]),
        device=device,
        freeze_kind=True,
    )
    CKPT_PTR.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), CKPT_PTR)
    policy.eval()

    models = {
        "baseline_joint": load_policy(CKPT_BASE, device),
        "generic_joint": load_policy(CKPT_GENERIC, device),
        "node_only": policy,
    }
    online = {}
    offline = {}
    for name, pol in models.items():
        if pol is None:
            continue
        online[name] = {}
        offline[name] = {}
        for n_dev in (6, 8, 10):
            print(f"=== Forced-cue eval {name} N={n_dev} ===")
            online[name][n_dev] = forced_cue_rank(
                pol, protocol, n_devices=n_dev, n=n_ep, seed_base=seed + 100 * n_dev, device=device
            )
            offline[name][n_dev] = rule_misconfig_pointer_eval(
                pol,
                protocol,
                n_devices=n_dev,
                n_ok=30,
                seed0=500_000 + 2000 * n_dev,
                device=device,
            )
            r = online[name][n_dev]["first_cue_spoke"]
            print(
                f"  spoke cue n={r.get('n', 0)} top1={_pct(r.get('top1'))} top3={_pct(r.get('top3'))} "
                f"rank={_f(r.get('mean_rank'))}"
            )

    # Decision
    no10 = online.get("node_only", {}).get(10, {}).get("first_cue_spoke") or {}
    gj10 = online.get("generic_joint", {}).get(10, {}).get("first_cue_spoke") or {}
    t1 = no10.get("top1") or 0
    t3 = no10.get("top3") or 0
    if t1 >= 0.70:
        decision = "pointer_learnable_joint_was_bottleneck"
        claim = (
            "Node-only training recovers spoke cue→GT @N10 — binding is learnable; "
            "joint multi-head BC was the bottleneck."
        )
    elif t1 >= 0.35 or t3 >= 0.70:
        decision = "pointer_partial_scale_gap"
        claim = (
            "Node-only improves over joint but still incomplete at N=10 — "
            "capacity/curriculum still needed for full binding."
        )
    else:
        decision = "pointer_representation_limit"
        claim = (
            "Even node-only fails spoke binding @N10 — graph features / MP do not carry "
            "asym-missing→endpoint correspondence under scale."
        )

    lines = [
        "# Pointer-only supervised probe",
        "",
        "> Train: Rule node steps (status/iface/node-diagnose) on Star{4,5,6}; freeze kind/edge/fault heads.",
        "> Eval: **forced-scan** first-cue GT node rank (kind-independent) + Rule-state ranks.",
        f"> Decision: **{decision}**",
        "",
        claim,
        "",
        f"- ckpt: `{CKPT_PTR.as_posix()}` last_loss={losses[-1]:.4f} n_train={len(labs_n)}",
        "",
        "## Online first-cue — all GT",
        "",
        "| Model | N | top-1 | top-3 | mean-rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("baseline_joint", "generic_joint", "node_only"):
        if name not in online:
            continue
        for n_dev in (6, 8, 10):
            r = online[name][n_dev]["first_cue_gt"]
            lines.append(
                f"| {name} | {n_dev} | {_pct(r.get('top1'))} | {_pct(r.get('top3'))} | {_f(r.get('mean_rank'))} |"
            )

    lines.extend(
        [
            "",
            "## Online first-cue — **spoke only** (critical)",
            "",
            "| Model | N | top-1 | top-3 | mean-rank |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("baseline_joint", "generic_joint", "node_only"):
        if name not in online:
            continue
        for n_dev in (6, 8, 10):
            r = online[name][n_dev]["first_cue_spoke"]
            lines.append(
                f"| {name} | {n_dev} | {_pct(r.get('top1'))} | {_pct(r.get('top3'))} | {_f(r.get('mean_rank'))} |"
            )

    lines.extend(
        [
            "",
            "## Offline Rule iface — pred=GT",
            "",
            "| Model | N | pred=GT | GT top-1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("baseline_joint", "generic_joint", "node_only"):
        if name not in offline:
            continue
        for n_dev in (6, 8, 10):
            o = offline[name][n_dev]
            iface = o.get("at_expert_iface") or {}
            lines.append(
                f"| {name} | {n_dev} | {_pct(o.get('pred_is_gt_when_iface'))} | {_pct(iface.get('top1'))} |"
            )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Spoke@N10: Generic joint top-1={_pct(gj10.get('top1'))} → node-only top-1={_pct(t1)} top-3={_pct(t3)}",
            "",
            claim,
            "",
            "If learnable: add node-pointer curriculum / loss weight on status·iface after missing cues.",
            "If representation limit: enrich edge→endpoint features (e.g. missing-incident flags on nodes).",
            "",
            "Run: `python -m src.run_p1_pointer_only_probe`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "p1_pointer_only_probe.md").write_text("\n".join(lines), encoding="utf-8")
    (results / "p1_pointer_only_probe.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "claim": claim,
                "losses": losses,
                "n_train": len(labs_n),
                "online": online,
                "offline": offline,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(decision)
    print(claim)
    print("Wrote", reports / "p1_pointer_only_probe.md")


if __name__ == "__main__":
    main()
