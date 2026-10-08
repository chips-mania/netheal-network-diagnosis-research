"""Commit ablation: Baseline vs Generic Commit vs Current (fault-hard ×2).

Generic = diagnose oversample ×3 + w_diag=4, w_hard=1 (no Link/Misconfig boost).
Current numbers reused from prior `p1_commit_fix` run when available.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch

from .graph_policy import KIND_ORDER, GraphPointerPolicy
from .protocol import PROJECT_ROOT, load_protocol
from .run_graph_expert_bc import collect_rule_v2_graph, evaluate_graph_policy
from .run_p1_commit_fix import (
    CKPT_BASE,
    CKPT_FIX,
    DIAG_I,
    oversample_diagnose,
    train_commit_bc,
)
from .run_star_scale_poc import _pf
from .run_symmetric_reward_poc import set_seeds

CKPT_GENERIC = PROJECT_ROOT / "models" / "checkpoints" / "graph_pointer_p1_commit_generic.pt"


def _exact_pf(block: Dict, ft: str) -> float:
    pf = (block.get("per_fault") or {}).get(ft) or {}
    if "dsr" in pf:
        return float(pf["dsr"])
    n = int(pf.get("n") or 0)
    return float(pf.get("exact_n") or 0) / max(1, n)


def main() -> None:
    protocol = load_protocol()
    cfg = protocol["graph_expert_bc_poc"]
    set_seeds(int(cfg["policy_seed"]))
    device = torch.device("cpu")
    pad = int(cfg["pad_max_devices"])
    demos_per = int(cfg["demos_per_n"])
    oversample = int(cfg.get("commit_oversample", 3))
    w_diag = float(cfg.get("commit_w_diag", 4.0))
    epochs = int(cfg.get("commit_epochs", 40))
    n_eval = int(cfg["eval_episodes"])
    seed = int(cfg["eval_seed_base"])

    # --- collect once ---
    graphs_all, labs_all = [], []
    for n in cfg["train_ns"]:
        n = int(n)
        print(f"=== Collect Rule-v2 graph N={n} ===")
        g, lab, gst = collect_rule_v2_graph(
            protocol, n_devices=n, n_ok=demos_per, seed0=80_000 + 1000 * n
        )
        print(" ", gst)
        graphs_all.extend(g)
        labs_all.extend(lab)
    graphs_os, labs_os = oversample_diagnose(graphs_all, labs_all, copies=oversample)
    print(f"=== Generic train: {len(labs_all)} → {len(labs_os)} w_diag={w_diag} w_hard=1.0 ===")

    policy = GraphPointerPolicy(hidden=64, layers=2).to(device)
    if CKPT_BASE.is_file():
        policy.load_state_dict(torch.load(CKPT_BASE, map_location=device))
        print(f"Warm-start {CKPT_BASE}")

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
        w_hard=1.0,
    )
    CKPT_GENERIC.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), CKPT_GENERIC)

    generic: Dict[int, Dict] = {}
    for n in (6, 8, 10):
        print(f"=== Eval generic-commit star_{n} ===")
        generic[n] = evaluate_graph_policy(
            policy,
            protocol,
            topology="star",
            n_devices=n,
            n=n_eval,
            seed_base=seed + 200 * n,
            device=device,
        )
        print(f"  exact={generic[n]['exact']:.0%} timeout={generic[n]['timeout_rate']:.0%}")

    # Baseline from Phase3 report; Current from commit_fix json if present
    baseline_exact = {6: 1.00, 8: 0.90, 10: 0.42}
    baseline_to = {6: 0.0, 8: 0.10, 10: 0.58}
    baseline_pf = {
        10: {
            "device_failure": 0.68,
            "performance_degradation": 1.0,
            "misconfiguration": 0.0,
            "link_failure": 0.0,
        },
        8: {
            "device_failure": 1.0,
            "performance_degradation": 0.96,
            "misconfiguration": 0.64,
            "link_failure": 1.0,
        },
    }

    current_exact = {6: 0.99, 8: 0.96, 10: 0.67}
    current_to = {6: 0.0, 8: 0.04, 10: 0.33}
    current_pf = {
        10: {
            "device_failure": 0.68,
            "performance_degradation": 0.92,
            "misconfiguration": 0.08,
            "link_failure": 1.0,
        },
        8: {
            "device_failure": 1.0,
            "performance_degradation": 0.96,
            "misconfiguration": 0.88,
            "link_failure": 1.0,
        },
    }
    fix_json = PROJECT_ROOT / "results" / "p1_commit_fix.json"
    if fix_json.is_file():
        prev = json.loads(fix_json.read_text(encoding="utf-8"))
        for k, v in (prev.get("exact") or {}).items():
            current_exact[int(k)] = float(v)
        for k, v in (prev.get("timeout") or {}).items():
            current_to[int(k)] = float(v)
        for k, block in (prev.get("per_fault") or {}).items():
            if not block:
                continue
            current_pf[int(k)] = {ft: float(b.get("dsr", 0)) for ft, b in block.items()}

    g_exact = {n: float(generic[n]["exact"]) for n in (6, 8, 10)}
    g_to = {n: float(generic[n]["timeout_rate"]) for n in (6, 8, 10)}

    # Decision rules
    link10_g = _exact_pf(generic[10], "link_failure")
    link10_c = current_pf.get(10, {}).get("link_failure", 0.0)
    close_n10 = abs(g_exact[10] - current_exact[10]) <= 0.08
    link_recovered = link10_g >= 0.80
    if link_recovered and close_n10 and g_exact[10] >= 0.55:
        decision = "generic_commit_sufficient"
        claim = (
            "Generic diagnose-commit objective (no fault-specific ×2) recovers Link and "
            "matches Current within noise — fault-hard weight not required for the main gain."
        )
    elif link_recovered and g_exact[10] >= baseline_exact[10] + 0.10:
        decision = "generic_helps_but_hard_adds_more"
        claim = (
            "Generic commit helps (esp. Link), but Current hard×2 still adds measurable Exact; "
            "report both generic gain and fault-specific residual."
        )
    elif not link_recovered and link10_c >= 0.80:
        decision = "hard_weight_necessary"
        claim = (
            "Link recovery appears to need Link/Misconfig diagnose upweight; "
            "generic commit alone insufficient."
        )
    else:
        decision = "inconclusive"
        claim = "Compare tables; effect sizes unclear."

    lines = [
        "# Commit ablation: Baseline vs Generic vs Current",
        "",
        "> Baseline: plain P1 BC.  ",
        "> **Generic**: warm-start + diagnose ×3 + w_diag=4, **w_hard=1**.  ",
        "> **Current**: same + **w_hard=2** on misconfig/link diagnose (prior run).",
        f"> Decision: **{decision}**",
        "",
        claim,
        "",
        f"- Generic ckpt: `{CKPT_GENERIC.as_posix()}` last loss={losses[-1]:.4f}",
        "",
        "## Exact / Timeout",
        "",
        "| N | Baseline Exact | Generic | Current | Generic TO | Current TO |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n in (6, 8, 10):
        lines.append(
            f"| {n} | {baseline_exact[n]:.0%} | {g_exact[n]:.0%} | {current_exact[n]:.0%} | "
            f"{g_to[n]:.0%} | {current_to[n]:.0%} |"
        )

    lines.extend(
        [
            "",
            "## Per-fault Exact @ N=8",
            "",
            "| Fault | Baseline | Generic | Current |",
            "|---|---:|---:|---:|",
        ]
    )
    for ft in (
        "device_failure",
        "performance_degradation",
        "misconfiguration",
        "link_failure",
    ):
        lines.append(
            f"| {ft} | {baseline_pf[8].get(ft, 0):.0%} | {_exact_pf(generic[8], ft):.0%} | "
            f"{current_pf[8].get(ft, 0):.0%} |"
        )

    lines.extend(
        [
            "",
            "## Per-fault Exact @ N=10",
            "",
            "| Fault | Baseline | Generic | Current |",
            "|---|---:|---:|---:|",
        ]
    )
    for ft in (
        "device_failure",
        "performance_degradation",
        "misconfiguration",
        "link_failure",
    ):
        lines.append(
            f"| {ft} | {baseline_pf[10].get(ft, 0):.0%} | {_exact_pf(generic[10], ft):.0%} | "
            f"{current_pf[10].get(ft, 0):.0%} |"
        )

    lines.extend(
        [
            "",
            "## Generic per-fault detail",
            "",
        ]
    )
    for n in (6, 8, 10):
        lines.append(f"### star_{n}")
        v = generic[n]
        lines.append(f"- Device {_pf(v, 'device_failure')}")
        lines.append(f"- Perf {_pf(v, 'performance_degradation')}")
        lines.append(f"- Misconfig {_pf(v, 'misconfiguration')}")
        lines.append(f"- Link {_pf(v, 'link_failure')}")
        lines.append("")

    lines.extend(
        [
            "## Go / claim",
            "",
            f"- decision: **{decision}**",
            f"- Link@N10 Generic={link10_g:.0%} Current={link10_c:.0%}",
            f"- N10 Exact Generic={g_exact[10]:.0%} Current={current_exact[10]:.0%} Baseline={baseline_exact[10]:.0%}",
            "",
            "Run: `python -m src.run_p1_commit_ablation`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "p1_commit_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "decision": decision,
        "claim": claim,
        "losses": losses,
        "generic_exact": {str(k): v for k, v in g_exact.items()},
        "generic_timeout": {str(k): v for k, v in g_to.items()},
        "generic_per_fault": {str(n): generic[n].get("per_fault") for n in (6, 8, 10)},
        "baseline_exact": {str(k): v for k, v in baseline_exact.items()},
        "current_exact": {str(k): v for k, v in current_exact.items()},
        "recipe_generic": {"oversample": oversample, "w_diag": w_diag, "w_hard": 1.0, "epochs": epochs},
    }
    (results / "p1_commit_ablation.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"decision={decision}")
    print(claim)
    print("Wrote", reports / "p1_commit_ablation.md")


if __name__ == "__main__":
    main()
