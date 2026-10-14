"""P1 training failure audit: tiny overfit → per-head → P0 GNN+flat.

Sequential diagnostics before scaling epochs / DAgger.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env_utils import make_masked_base_env
from .graph_obs import NODE_DIM, EDGE_DIM, build_graph_obs
from .graph_policy import (
    FAULT_ORDER,
    KIND_ORDER,
    GraphPointerPolicy,
    bc_loss,
    collate_graphs,
    pointer_to_labels,
)
from .pointer_actions import encode_flat_as_pointer
from .protocol import PROJECT_ROOT, load_protocol
from .run_graph_expert_bc import collect_rule_v2_flat, collect_rule_v2_graph
from .run_star_scale_poc import _env_kwargs
from .run_symmetric_reward_poc import set_seeds
from .action_mask import base_env
from .action_trace import _diagnosis_correct, _parse_action_spec
from .rule_agent_v2 import RuleAgentV2
from .star_pad import steps_budget


class GraphFlatPolicy(nn.Module):
    """P0: GNN encoder + flat Discrete head (pad action space)."""

    def __init__(self, n_actions: int, hidden: int = 64, layers: int = 2, global_dim: int = 6):
        super().__init__()
        self.backbone = GraphPointerPolicy(hidden=hidden, layers=layers, global_dim=global_dim)
        self.n_actions = n_actions
        self.action_head = nn.Linear(hidden, n_actions)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        h, he, g = self.backbone.encode(
            batch["node_feat"],
            batch["edge_index"],
            batch["edge_feat"],
            batch["node_mask"],
            batch["edge_mask"],
            batch["globals"],
        )
        return self.action_head(g)


def collect_graph_flat_pairs(
    protocol, *, n_devices: int, pad: int, n_ok: int, seed0: int, max_trans: int | None = None
) -> Tuple[List, List[int], Dict]:
    """(GraphObs, flat_action_id) under star pad — for P0."""
    from .action_mask import action_masks_for_env

    kw = _env_kwargs(protocol, n_devices, pad)
    max_steps = kw["max_episode_steps"]
    env = make_masked_base_env(["star"], protocol=protocol, **kw)
    graphs, acts = [], []
    ok = tries = 0
    per: Dict[str, int] = {}
    while ok < n_ok and tries < n_ok * 8:
        if max_trans is not None and len(acts) >= max_trans:
            break
        tries += 1
        agent = RuleAgentV2()
        obs, info = env.reset(seed=seed0 + tries)
        root = base_env(env)
        gt = (info.get("ground_truth_fault") or {}).get("type", "unknown")
        success = False
        ep_g, ep_a = [], []
        for t in range(1, max_steps + 1):
            g = build_graph_obs(env, mem=agent.mem)
            try:
                action = int(agent.choose(env, obs, t, []))
            except RuntimeError:
                success = False
                break
            ep_g.append(g)
            ep_a.append(action)
            obs, reward, term, trunc, info = env.step(action)
            agent.observe_step(info)
            spec = _parse_action_spec(info, root, action)
            if spec is not None and _diagnosis_correct(info, spec) is True:
                success = True
            if term or trunc:
                break
        if success and ep_g:
            ok += 1
            per[gt] = per.get(gt, 0) + 1
            graphs.extend(ep_g)
            acts.extend(ep_a)
    n_actions = int(base_env(env).action_space.n)
    env.close()
    return graphs, acts, {
        "ok": ok,
        "n_transitions": len(acts),
        "n_actions": n_actions,
        "per_fault_ok": per,
        "n_devices": n_devices,
    }


def _train_p1(
    policy: GraphPointerPolicy,
    graphs,
    labels,
    *,
    epochs: int,
    batch: int,
    lr: float,
    pad_nodes: int,
    pad_edges: int,
    device: torch.device,
) -> List[float]:
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(labels)
    losses = []
    policy.train()
    for ep in range(epochs):
        idx = np.random.permutation(n)
        ep_loss = []
        for start in range(0, n, batch):
            sel = idx[start : start + batch]
            if len(sel) < 1:
                continue
            # allow batch=1 for tiny
            gs = [graphs[i] for i in sel]
            bt = collate_graphs(gs, pad_nodes=pad_nodes, pad_edges=pad_edges)
            bt = {k: v.to(device) for k, v in bt.items()}
            lab = {
                "kind": torch.tensor([labels[i]["kind"] for i in sel], device=device),
                "node": torch.tensor([labels[i]["node"] for i in sel], device=device),
                "edge": torch.tensor([labels[i]["edge"] for i in sel], device=device),
                "fault": torch.tensor([labels[i]["fault"] for i in sel], device=device),
            }
            logits = policy.forward_logits(
                bt["node_feat"], bt["edge_index"], bt["edge_feat"], bt["node_mask"], bt["edge_mask"], bt["globals"]
            )
            loss = bc_loss(logits, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss.append(float(loss.item()))
        losses.append(float(np.mean(ep_loss)) if ep_loss else 0.0)
    return losses


@torch.no_grad()
def eval_p1_heads(policy, graphs, labels, *, pad_nodes, pad_edges, device) -> Dict:
    policy.eval()
    kind_c = kind_t = 0
    node_c = node_t = 0
    edge_c = edge_t = 0
    fault_c = fault_t = 0
    by_kind = defaultdict(lambda: {"n": 0, "kind_ok": 0})
    # joint: all required heads correct
    joint_ok = joint_n = 0
    bs = 32
    for start in range(0, len(labels), bs):
        sel = list(range(start, min(start + bs, len(labels))))
        gs = [graphs[i] for i in sel]
        bt = collate_graphs(gs, pad_nodes=pad_nodes, pad_edges=pad_edges)
        bt = {k: v.to(device) for k, v in bt.items()}
        logits = policy.forward_logits(
            bt["node_feat"], bt["edge_index"], bt["edge_feat"], bt["node_mask"], bt["edge_mask"], bt["globals"]
        )
        pred_k = logits["kind"].argmax(-1).cpu().numpy()
        pred_n = logits["node"].argmax(-1).cpu().numpy()
        pred_e = logits["edge"].argmax(-1).cpu().numpy()
        pred_f = logits["fault"].argmax(-1).cpu().numpy()
        for j, i in enumerate(sel):
            lab = labels[i]
            k = lab["kind"]
            kind_t += 1
            kind_ok = int(pred_k[j] == k)
            kind_c += kind_ok
            by_kind[KIND_ORDER[k].value]["n"] += 1
            by_kind[KIND_ORDER[k].value]["kind_ok"] += kind_ok
            ok = kind_ok
            if lab["node"] >= 0:
                node_t += 1
                nok = int(pred_n[j] == lab["node"])
                node_c += nok
                ok = ok and nok
            if lab["edge"] >= 0:
                edge_t += 1
                eok = int(pred_e[j] == lab["edge"])
                edge_c += eok
                ok = ok and eok
            if lab["fault"] >= 0:
                fault_t += 1
                fok = int(pred_f[j] == lab["fault"])
                fault_c += fok
                ok = ok and fok
            joint_n += 1
            joint_ok += int(ok)
    return {
        "kind_acc": kind_c / max(1, kind_t),
        "node_acc": node_c / max(1, node_t) if node_t else None,
        "edge_acc": edge_c / max(1, edge_t) if edge_t else None,
        "fault_acc": fault_c / max(1, fault_t) if fault_t else None,
        "joint_required_acc": joint_ok / max(1, joint_n),
        "counts": {"kind": kind_t, "node": node_t, "edge": edge_t, "fault": fault_t},
        "by_kind": {k: {"n": v["n"], "kind_acc": v["kind_ok"] / max(1, v["n"])} for k, v in by_kind.items()},
    }


def _train_p0(policy: GraphFlatPolicy, graphs, acts, *, epochs, batch, lr, pad_nodes, pad_edges, device):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(acts)
    losses = []
    policy.train()
    for ep in range(epochs):
        idx = np.random.permutation(n)
        ep_loss = []
        for start in range(0, n, batch):
            sel = idx[start : start + batch]
            if len(sel) < 1:
                continue
            gs = [graphs[i] for i in sel]
            bt = collate_graphs(gs, pad_nodes=pad_nodes, pad_edges=pad_edges)
            bt = {k: v.to(device) for k, v in bt.items()}
            y = torch.tensor([acts[i] for i in sel], device=device, dtype=torch.long)
            logits = policy(bt)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss.append(float(loss.item()))
        losses.append(float(np.mean(ep_loss)) if ep_loss else 0.0)
    return losses


@torch.no_grad()
def eval_p0_acc(policy, graphs, acts, *, pad_nodes, pad_edges, device) -> float:
    policy.eval()
    correct = total = 0
    bs = 32
    for start in range(0, len(acts), bs):
        sel = list(range(start, min(start + bs, len(acts))))
        bt = collate_graphs([graphs[i] for i in sel], pad_nodes=pad_nodes, pad_edges=pad_edges)
        bt = {k: v.to(device) for k, v in bt.items()}
        pred = policy(bt).argmax(-1).cpu().numpy()
        for j, i in enumerate(sel):
            total += 1
            correct += int(pred[j] == acts[i])
    return correct / max(1, total)


def main() -> None:
    protocol = load_protocol()
    set_seeds(1)
    device = torch.device("cpu")
    pad_nodes, pad_edges = 12, 132
    pad = 12

    print("=== Collect graph labels N=6 (for audit) ===")
    graphs, labels, gst = collect_rule_v2_graph(protocol, n_devices=6, n_ok=400, seed0=90_000)
    print(gst)
    assert len(labels) >= 200, gst

    # --- Step 1: tiny overfit ---
    tiny_n = 200
    tiny_g, tiny_l = graphs[:tiny_n], labels[:tiny_n]
    print(f"=== Step1 tiny overfit n={tiny_n} epochs=300 ===")
    p_tiny = GraphPointerPolicy(hidden=64, layers=2).to(device)
    losses_tiny = _train_p1(
        p_tiny, tiny_g, tiny_l, epochs=300, batch=32, lr=1e-3,
        pad_nodes=pad_nodes, pad_edges=pad_edges, device=device,
    )
    heads_tiny = eval_p1_heads(p_tiny, tiny_g, tiny_l, pad_nodes=pad_nodes, pad_edges=pad_edges, device=device)
    print(f"  last_loss={losses_tiny[-1]:.4f} joint={heads_tiny['joint_required_acc']:.1%} kind={heads_tiny['kind_acc']:.1%}")

    tiny_ok = losses_tiny[-1] < 0.05 and heads_tiny["joint_required_acc"] >= 0.98
    tiny_soft = losses_tiny[-1] < 0.2 and heads_tiny["joint_required_acc"] >= 0.90

    # --- Step 2: per-head on larger train subset (same as Phase3 scale sample) ---
    print("=== Step2 per-head on full collected set after more training ===")
    # Train longer on all collected (~2k–4k) to see which heads lag
    p_full = GraphPointerPolicy(hidden=64, layers=2).to(device)
    losses_full = _train_p1(
        p_full, graphs, labels, epochs=80, batch=64, lr=3e-4,
        pad_nodes=pad_nodes, pad_edges=pad_edges, device=device,
    )
    heads_full = eval_p1_heads(p_full, graphs, labels, pad_nodes=pad_nodes, pad_edges=pad_edges, device=device)
    print(
        f"  last_loss={losses_full[-1]:.4f} joint={heads_full['joint_required_acc']:.1%} "
        f"node={heads_full['node_acc']} edge={heads_full['edge_acc']} fault={heads_full['fault_acc']}"
    )

    # --- Step 3: P0 ---
    print("=== Step3 P0 GNN+flat collect+train ===")
    g0, a0, st0 = collect_graph_flat_pairs(
        protocol, n_devices=6, pad=pad, n_ok=400, seed0=91_000, max_trans=None
    )
    # align length with tiny/full for fair-ish loss compare: use all
    n_actions = st0["n_actions"]
    print(" ", st0)
    # tiny P0
    tiny_m = min(200, len(a0))
    p0_tiny = GraphFlatPolicy(n_actions=n_actions, hidden=64, layers=2).to(device)
    losses_p0_tiny = _train_p0(
        p0_tiny, g0[:tiny_m], a0[:tiny_m], epochs=300, batch=32, lr=1e-3,
        pad_nodes=pad_nodes, pad_edges=pad_edges, device=device,
    )
    acc_p0_tiny = eval_p0_acc(p0_tiny, g0[:tiny_m], a0[:tiny_m], pad_nodes=pad_nodes, pad_edges=pad_edges, device=device)
    print(f"  P0 tiny last_loss={losses_p0_tiny[-1]:.4f} acc={acc_p0_tiny:.1%}")

    p0_full = GraphFlatPolicy(n_actions=n_actions, hidden=64, layers=2).to(device)
    losses_p0 = _train_p0(
        p0_full, g0, a0, epochs=80, batch=64, lr=3e-4,
        pad_nodes=pad_nodes, pad_edges=pad_edges, device=device,
    )
    acc_p0 = eval_p0_acc(p0_full, g0, a0, pad_nodes=pad_nodes, pad_edges=pad_edges, device=device)
    print(f"  P0 full last_loss={losses_p0[-1]:.4f} acc={acc_p0:.1%}")

    # Diagnosis
    if tiny_ok:
        arch = "architecture_can_overfit"
    elif tiny_soft:
        arch = "architecture_mostly_ok_slow_overfit"
    else:
        arch = "architecture_or_label_bug"

    if losses_p0_tiny[-1] < 0.1 and acc_p0_tiny >= 0.95:
        p0_tiny_ok = True
    else:
        p0_tiny_ok = False

    if p0_tiny_ok and not tiny_ok:
        bottleneck = "pointer_multihead"
    elif not p0_tiny_ok and not tiny_ok:
        bottleneck = "gnn_or_graph_obs_or_shared"
    elif tiny_ok and heads_full["joint_required_acc"] < 0.85:
        bottleneck = "optimization_or_data_scale_not_arch"
    else:
        bottleneck = "unclear_or_ok"

    # Recommend next
    if arch == "architecture_or_label_bug":
        next_step = "Fix pointer labels / conditional loss / masking before more data"
    elif bottleneck == "pointer_multihead":
        next_step = "Simplify pointer heads / stronger kind curriculum / check Misconfig labels; then retrain P1"
    elif bottleneck == "gnn_or_graph_obs_or_shared":
        next_step = "Inspect graph_obs features / GNN capacity; P0 also fails tiny overfit"
    else:
        next_step = "Increase P1 capacity/epochs with fixed recipe; target Seen≥90%"

    payload = {
        "step1_tiny_overfit": {
            "n": tiny_n,
            "epochs": 300,
            "last_loss": losses_tiny[-1],
            "loss_curve_tail": losses_tiny[-5:],
            "heads": heads_tiny,
            "pass_strict": tiny_ok,
            "pass_soft": tiny_soft,
            "arch_readout": arch,
        },
        "step2_per_head_full_subset": {
            "n_transitions": len(labels),
            "epochs": 80,
            "last_loss": losses_full[-1],
            "heads": heads_full,
        },
        "step3_p0": {
            "n_actions": n_actions,
            "n_transitions": len(a0),
            "tiny": {"n": tiny_m, "last_loss": losses_p0_tiny[-1], "acc": acc_p0_tiny, "ok": p0_tiny_ok},
            "full": {"last_loss": losses_p0[-1], "acc": acc_p0},
        },
        "bottleneck": bottleneck,
        "next_step": next_step,
    }

    lines = [
        "# P1 train audit (tiny → heads → P0)",
        "",
        f"> Bottleneck read: **{bottleneck}**",
        f"> Next: {next_step}",
        "",
        "## Step 1 — Tiny overfit (200 transitions, 300 ep)",
        "",
        f"- Last loss: **{losses_tiny[-1]:.4f}**",
        f"- Joint required-head acc: **{heads_tiny['joint_required_acc']:.1%}**",
        f"- Kind / Node / Edge / Fault: "
        f"{heads_tiny['kind_acc']:.1%} / {heads_tiny['node_acc']} / {heads_tiny['edge_acc']} / {heads_tiny['fault_acc']}",
        f"- Strict pass (loss<0.05 & joint≥98%): **{tiny_ok}**",
        f"- Soft pass: **{tiny_soft}** → `{arch}`",
        "",
        "### By kind (tiny)",
        "",
    ]
    for k, v in sorted(heads_tiny["by_kind"].items()):
        lines.append(f"- {k}: n={v['n']}, kind_acc={v['kind_acc']:.1%}")

    lines.extend(
        [
            "",
            "## Step 2 — Per-head on ~N=6 Rule-v2 set (80 ep)",
            "",
            f"- Transitions: {len(labels)}, last loss **{losses_full[-1]:.4f}**",
            f"- Joint: **{heads_full['joint_required_acc']:.1%}**",
            f"- Kind: {heads_full['kind_acc']:.1%}",
            f"- Node pointer: {heads_full['node_acc']}",
            f"- Edge pointer: {heads_full['edge_acc']}",
            f"- Fault type: {heads_full['fault_acc']}",
            "",
            "### By kind",
            "",
        ]
    )
    for k, v in sorted(heads_full["by_kind"].items()):
        lines.append(f"- {k}: n={v['n']}, kind_acc={v['kind_acc']:.1%}")

    lines.extend(
        [
            "",
            "## Step 3 — P0 GNN+flat",
            "",
            f"- Action space n={n_actions}",
            f"- Tiny(200): loss **{losses_p0_tiny[-1]:.4f}**, acc **{acc_p0_tiny:.1%}**, ok={p0_tiny_ok}",
            f"- Full(~{len(a0)}): loss **{losses_p0[-1]:.4f}**, acc **{acc_p0:.1%}**",
            "",
            "## Readout",
            "",
            "| Check | Result |",
            "|---|---|",
            f"| P1 tiny overfit | {'PASS' if tiny_ok else ('SOFT' if tiny_soft else 'FAIL')} |",
            f"| P0 tiny overfit | {'PASS' if p0_tiny_ok else 'FAIL'} |",
            f"| Bottleneck | {bottleneck} |",
            "",
            "Run: `python -m src.run_p1_train_audit`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "p1_train_audit.md").write_text("\n".join(lines), encoding="utf-8")
    (results / "p1_train_audit.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("bottleneck=", bottleneck)
    print("Wrote", reports / "p1_train_audit.md")


if __name__ == "__main__":
    main()
