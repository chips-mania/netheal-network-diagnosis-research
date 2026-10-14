"""P1 commit fix: oversample diagnose + weighted kind CE, re-eval Star 6/8/10.

Does not invent GT labels — only upweights expert diagnose transitions so the
policy learns “evidence ready → submit” more strongly under scale shift.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import torch

from .graph_policy import KIND_ORDER, GraphPointerPolicy, bc_loss, collate_graphs
from .protocol import PROJECT_ROOT, load_protocol
from .run_graph_expert_bc import collect_rule_v2_graph, evaluate_graph_policy
from .run_p1_commit_margin_audit import rollout_margin
from .run_star_scale_poc import _pf
from .run_symmetric_reward_poc import set_seeds

DIAG_I = next(i for i, k in enumerate(KIND_ORDER) if k.value == "diagnose")
# FAULT_ORDER: device, perf, misconfig, link
MISCONFIG_I = 2
LINK_I = 3

CKPT_BASE = PROJECT_ROOT / "models" / "checkpoints" / "graph_pointer_p1.pt"
CKPT_FIX = PROJECT_ROOT / "models" / "checkpoints" / "graph_pointer_p1_commit.pt"


def oversample_diagnose(
    graphs: List, labels: List[Dict[str, int]], *, copies: int
) -> Tuple[List, List[Dict[str, int]]]:
    if copies <= 1:
        return graphs, labels
    extra_g, extra_l = [], []
    for g, lab in zip(graphs, labels):
        if lab["kind"] != DIAG_I:
            continue
        for _ in range(copies - 1):
            extra_g.append(g)
            extra_l.append(lab)
    return graphs + extra_g, labels + extra_l


def kind_row_weights(labels_batch: torch.Tensor, fault_batch: torch.Tensor, *, w_diag: float, w_hard: float) -> torch.Tensor:
    """Per-row kind CE weights. Hard = misconfig/link diagnose."""
    w = torch.ones(labels_batch.shape[0], device=labels_batch.device)
    is_diag = labels_batch == DIAG_I
    w = torch.where(is_diag, w * w_diag, w)
    hard = is_diag & ((fault_batch == MISCONFIG_I) | (fault_batch == LINK_I))
    w = torch.where(hard, w * w_hard, w)
    return w


def train_commit_bc(
    policy: GraphPointerPolicy,
    graphs: List,
    labels: List[Dict[str, int]],
    *,
    epochs: int,
    batch: int,
    lr: float,
    pad_nodes: int,
    pad_edges: int,
    device: torch.device,
    w_diag: float,
    w_hard: float,
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
            if len(sel) < 2:
                continue
            gs = [graphs[i] for i in sel]
            batch_t = collate_graphs(gs, pad_nodes=pad_nodes, pad_edges=pad_edges)
            batch_t = {k: v.to(device) for k, v in batch_t.items()}
            lab = {
                "kind": torch.tensor([labels[i]["kind"] for i in sel], device=device),
                "node": torch.tensor([labels[i]["node"] for i in sel], device=device),
                "edge": torch.tensor([labels[i]["edge"] for i in sel], device=device),
                "fault": torch.tensor([labels[i]["fault"] for i in sel], device=device),
            }
            kw = kind_row_weights(lab["kind"], lab["fault"], w_diag=w_diag, w_hard=w_hard)
            logits = policy.forward_logits(
                batch_t["node_feat"],
                batch_t["edge_index"],
                batch_t["edge_feat"],
                batch_t["node_mask"],
                batch_t["edge_mask"],
                batch_t["globals"],
            )
            loss = bc_loss(logits, lab, kind_weight=kw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss.append(float(loss.item()))
        losses.append(float(np.mean(ep_loss)) if ep_loss else 0.0)
        if (ep + 1) % max(1, epochs // 4) == 0 or ep + 1 == epochs:
            print(f"  commit-bc epoch {ep+1}/{epochs} loss={losses[-1]:.4f}")
    return losses


def main() -> None:
    protocol = load_protocol()
    cfg = protocol["graph_expert_bc_poc"]
    set_seeds(int(cfg["policy_seed"]))
    device = torch.device("cpu")
    pad = int(cfg["pad_max_devices"])
    demos_per = int(cfg["demos_per_n"])
    w_diag = float(cfg.get("commit_w_diag", 4.0))
    w_hard = float(cfg.get("commit_w_hard", 2.0))
    oversample = int(cfg.get("commit_oversample", 3))
    epochs = int(cfg.get("commit_epochs", cfg.get("p1_epochs", 80)))

    graphs_all, labs_all = [], []
    collects = {}
    for n in cfg["train_ns"]:
        n = int(n)
        print(f"=== Collect Rule-v2 graph N={n} ===")
        g, lab, gst = collect_rule_v2_graph(
            protocol, n_devices=n, n_ok=demos_per, seed0=80_000 + 1000 * n
        )
        print(" ", gst)
        collects[f"graph_{n}"] = gst
        graphs_all.extend(g)
        labs_all.extend(lab)

    n_diag = sum(1 for lab in labs_all if lab["kind"] == DIAG_I)
    graphs_os, labs_os = oversample_diagnose(graphs_all, labs_all, copies=oversample)
    print(
        f"=== Commit train: base={len(labs_all)} diag={n_diag} "
        f"after_oversample×{oversample}={len(labs_os)} w_diag={w_diag} w_hard={w_hard} ==="
    )

    # Warm-start from baseline P1 if available
    policy = GraphPointerPolicy(hidden=64, layers=2).to(device)
    if CKPT_BASE.is_file():
        policy.load_state_dict(torch.load(CKPT_BASE, map_location=device))
        print(f"Warm-start from {CKPT_BASE}")
    else:
        print("No baseline ckpt — training from scratch")

    losses = train_commit_bc(
        policy,
        graphs_os,
        labs_os,
        epochs=epochs,
        batch=int(cfg["batch_size"]),
        lr=3e-4,
        pad_nodes=pad,
        pad_edges=int(cfg["pad_edges"]),
        device=device,
        w_diag=w_diag,
        w_hard=w_hard,
    )
    CKPT_FIX.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), CKPT_FIX)
    print(f"Saved {CKPT_FIX}")

    n_eval = int(cfg["eval_episodes"])
    seed = int(cfg["eval_seed_base"])
    p1_res = {}
    for n in (6, 8, 10):
        print(f"=== Eval commit-P1 star_{n} ===")
        p1_res[n] = evaluate_graph_policy(
            policy,
            protocol,
            topology="star",
            n_devices=n,
            n=n_eval,
            seed_base=seed + 200 * n,
            device=device,
        )
        print(
            f"  exact={p1_res[n]['exact']:.0%} timeout={p1_res[n]['timeout_rate']:.0%} "
            f"submit={p1_res[n]['submit_rate']:.0%}"
        )

    # Margin spot-check N=10 only (faster)
    print("=== Margin spot-check N=10 ===")
    margin10 = rollout_margin(
        policy,
        protocol,
        n_devices=10,
        n=n_eval,
        seed_base=seed + 9000,
        device=device,
        last_k=3,
    )

    # Baseline numbers from prior Phase3 report (fixed reference)
    baseline = {"6": 1.00, "8": 0.90, "10": 0.42}
    baseline_to = {"6": 0.0, "8": 0.10, "10": 0.58}

    lines = [
        "# P1 commit fix (diagnose oversample + weighted kind CE)",
        "",
        "> Goal: strengthen “evidence ready → diagnose” without GT shortcuts.",
        f"> Recipe: warm-start baseline P1; oversample diagnose ×{oversample}; "
        f"kind weight w_diag={w_diag}, extra ×{w_hard} on misconfig/link diagnose.",
        f"> Checkpoint: `{CKPT_FIX.as_posix()}`",
        "",
        f"- Transitions: {len(labs_all)} → {len(labs_os)} after oversample; last loss {losses[-1]:.4f}",
        "",
        "## Exact vs baseline P1",
        "",
        "| N | Baseline Exact | Commit Exact | Δ | Baseline TO | Commit TO |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n in (6, 8, 10):
        ex = float(p1_res[n]["exact"])
        to = float(p1_res[n]["timeout_rate"])
        b = baseline[str(n)]
        lines.append(
            f"| {n} | {b:.0%} | {ex:.0%} | {ex - b:+.0%} | {baseline_to[str(n)]:.0%} | {to:.0%} |"
        )

    lines.extend(["", "## Per-fault (commit P1)", ""])
    for n in (6, 8, 10):
        v = p1_res[n]
        lines.append(f"### star_{n}")
        if "per_fault" in v:
            lines.append(f"- Device {_pf(v, 'device_failure')}")
            lines.append(f"- Perf {_pf(v, 'performance_degradation')}")
            lines.append(f"- Misconfig {_pf(v, 'misconfiguration')}")
            lines.append(f"- Link {_pf(v, 'link_failure')}")
        lines.append("")

    cm = margin10["buckets"].get("conclude_match", {})
    tf = margin10["buckets"].get("timeout_final_step", {})
    tcm = margin10["buckets"].get("timeout_conclude_match", {})
    lines.extend(
        [
            "## N=10 commit margin (post-fix)",
            "",
            f"- Conclude-match: P(diag)={cm.get('p_diagnose_mean')} top-1={cm.get('diag_is_top')} "
            f"top-2={cm.get('diag_in_top2')} margin={cm.get('margin_mean')}",
            f"- Timeout final: P(diag)={tf.get('p_diagnose_mean')} top-1={tf.get('diag_is_top')} "
            f"kinds={tf.get('top_kind_hist')}",
            f"- Timeout∩ready: P(diag)={tcm.get('p_diagnose_mean')} top-2={tcm.get('diag_in_top2')}",
            "",
            "### Per-fault conclude-match @ N=10",
            "",
        ]
    )
    for ft, s in margin10.get("by_fault_conclude_match", {}).items():
        lines.append(
            f"- {ft}: n={s.get('n')} P(diag)={s.get('p_diagnose_mean')} "
            f"top-1={s.get('diag_is_top')} top-2={s.get('diag_in_top2')} margin={s.get('margin_mean')}"
        )

    ex10 = float(p1_res[10]["exact"])
    if ex10 >= 0.70:
        decision = "go_commit_helped"
    elif ex10 >= 0.55:
        decision = "partial_commit_help"
    else:
        decision = "commit_fix_insufficient"

    lines.extend(
        [
            "",
            f"## Decision: **{decision}**",
            "",
            "Run: `python -m src.run_p1_commit_fix`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "p1_commit_fix.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "decision": decision,
        "recipe": {"oversample": oversample, "w_diag": w_diag, "w_hard": w_hard, "epochs": epochs},
        "losses": losses,
        "exact": {str(n): float(p1_res[n]["exact"]) for n in (6, 8, 10)},
        "timeout": {str(n): float(p1_res[n]["timeout_rate"]) for n in (6, 8, 10)},
        "per_fault": {str(n): p1_res[n].get("per_fault") for n in (6, 8, 10)},
        "margin10": margin10,
        "collects": collects,
        "baseline_exact": baseline,
    }
    (results / "p1_commit_fix.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"decision={decision}")
    print("Wrote", reports / "p1_commit_fix.md")


if __name__ == "__main__":
    main()
