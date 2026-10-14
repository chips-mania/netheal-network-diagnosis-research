"""Commit factor ablation: weight-only (B) vs Generic, with ready/premature rates.

B = warm-start + w_diag=4, oversample×1, w_hard=1
Also scores Baseline & Generic checkpoints on the same ready/premature metrics.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch

from netheal.environment.actions import ActionCategory
from netheal.faults.injector import FaultType

from .action_trace import _base_env, _parse_action_spec
from .env_utils import make_masked_eval_env, metrics_env
from .graph_obs import build_graph_obs
from .graph_policy import KIND_ORDER, GraphPointerPolicy
from .pointer_actions import resolve_pointer
from .protocol import PROJECT_ROOT, load_protocol
from .rule_agent_v2 import RuleAgentV2
from .run_graph_expert_bc import collect_rule_v2_graph, evaluate_graph_policy
from .run_p1_commit_ablation import CKPT_GENERIC, _exact_pf
from .run_p1_commit_fix import CKPT_BASE, train_commit_bc
from .run_p1_star10_failure_audit import evidence_flags
from .run_star_scale_poc import _pf
from .run_symmetric_reward_poc import set_seeds
from .star_pad import steps_budget

DIAG_I = next(i for i, k in enumerate(KIND_ORDER) if k.value == "diagnose")
CKPT_B = PROJECT_ROOT / "models" / "checkpoints" / "graph_pointer_p1_commit_wonly.pt"


def commit_quality_rollout(
    policy: GraphPointerPolicy,
    protocol,
    *,
    n_devices: int,
    n: int,
    seed_base: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Ready commit vs premature diagnose rates (+ exact/timeout)."""
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    steps_max = steps_budget(n_devices, floor=int(protocol["max_episode_steps"]))
    ready_steps = ready_diag = 0
    diag_steps = premature_diag = 0
    ep_exact = ep_to = 0
    by_ft = {ft.value: {"n": 0, "exact": 0, "timeout": 0, "premature_ep": 0} for ft in faults}

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
            max_episode_steps=steps_max,
        )
        agent_ev = RuleAgentV2()
        base = _base_env(env)
        obs, info = env.reset(seed=seed_base + i)
        gt = info.get("ground_truth_fault") or {}
        ft = gt.get("type", "")
        floc = gt.get("location", "")
        agent_ev.reset()
        step = 0
        ep_premature = False

        while True:
            step += 1
            flags = evidence_flags(agent_ev.mem, ft, floc)
            ready = bool(flags.get("conclude_match"))
            g = build_graph_obs(env, mem=agent_ev.mem)
            ptr = policy.act(g, device)
            action_id = int(resolve_pointer(env, ptr))
            chose_diag = ptr.kind.value == "diagnose"

            if ready:
                ready_steps += 1
                if chose_diag:
                    ready_diag += 1
            if chose_diag:
                diag_steps += 1
                if not ready:
                    premature_diag += 1
                    ep_premature = True

            obs, reward, terminated, truncated, info = env.step(action_id)
            agent_ev.observe_step(info)
            agent_ev.refresh_plan(env)
            if terminated or truncated:
                break

        m = metrics_env(env).last_episode_metrics
        success = bool(m and m.diagnosis_success)
        timeout = (not success) and step >= steps_max
        if success:
            ep_exact += 1
        if timeout:
            ep_to += 1
        bucket = by_ft.setdefault(ft, {"n": 0, "exact": 0, "timeout": 0, "premature_ep": 0})
        bucket["n"] += 1
        bucket["exact"] += int(success)
        bucket["timeout"] += int(timeout)
        bucket["premature_ep"] += int(ep_premature)
        env.close()

    return {
        "n_devices": n_devices,
        "exact": ep_exact / max(1, n),
        "timeout": ep_to / max(1, n),
        "ready_commit_rate": ready_diag / max(1, ready_steps),
        "ready_steps": ready_steps,
        "premature_among_diag": premature_diag / max(1, diag_steps),
        "diag_steps": diag_steps,
        "premature_episode_rate": sum(b["premature_ep"] for b in by_ft.values()) / max(1, n),
        "by_fault": {
            k: {
                "n": v["n"],
                "exact": v["exact"] / max(1, v["n"]),
                "timeout": v["timeout"] / max(1, v["n"]),
                "premature_ep": v["premature_ep"] / max(1, v["n"]),
            }
            for k, v in by_ft.items()
        },
    }


def load_policy(path, device) -> Optional[GraphPointerPolicy]:
    if not path.is_file():
        return None
    p = GraphPointerPolicy(hidden=64, layers=2).to(device)
    p.load_state_dict(torch.load(path, map_location=device))
    p.eval()
    return p


def main() -> None:
    protocol = load_protocol()
    cfg = protocol["graph_expert_bc_poc"]
    set_seeds(int(cfg["policy_seed"]))
    device = torch.device("cpu")
    pad = int(cfg["pad_max_devices"])
    demos_per = int(cfg["demos_per_n"])
    w_diag = float(cfg.get("commit_w_diag", 4.0))
    epochs = int(cfg.get("commit_epochs", 40))
    n_eval = int(cfg["eval_episodes"])
    seed = int(cfg["eval_seed_base"])

    # --- train B ---
    graphs_all, labs_all = [], []
    for n in cfg["train_ns"]:
        n = int(n)
        print(f"=== Collect graph N={n} ===")
        g, lab, gst = collect_rule_v2_graph(
            protocol, n_devices=n, n_ok=demos_per, seed0=80_000 + 1000 * n
        )
        print(" ", gst)
        graphs_all.extend(g)
        labs_all.extend(lab)

    print(f"=== Train B weight-only: n={len(labs_all)} oversample×1 w_diag={w_diag} ===")
    policy_b = GraphPointerPolicy(hidden=64, layers=2).to(device)
    if CKPT_BASE.is_file():
        policy_b.load_state_dict(torch.load(CKPT_BASE, map_location=device))
        print(f"Warm-start {CKPT_BASE}")
    losses = train_commit_bc(
        policy_b,
        graphs_all,
        labs_all,
        epochs=epochs,
        batch=int(cfg["batch_size"]),
        lr=3e-4,
        pad_nodes=pad,
        pad_edges=int(cfg["pad_edges"]),
        device=device,
        w_diag=w_diag,
        w_hard=1.0,
    )
    CKPT_B.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_b.state_dict(), CKPT_B)
    policy_b.eval()

    # Standard exact eval for B
    b_eval = {}
    for n in (6, 8, 10):
        print(f"=== Eval B star_{n} ===")
        b_eval[n] = evaluate_graph_policy(
            policy_b,
            protocol,
            topology="star",
            n_devices=n,
            n=n_eval,
            seed_base=seed + 200 * n,
            device=device,
        )
        print(f"  exact={b_eval[n]['exact']:.0%} to={b_eval[n]['timeout_rate']:.0%}")

    # Commit-quality for Baseline / B / Generic on N=8,10
    models = {
        "baseline": load_policy(CKPT_BASE, device),
        "B_wonly": policy_b,
        "generic": load_policy(CKPT_GENERIC, device),
    }
    quality: Dict[str, Dict[int, Dict]] = {}
    q_seed = seed + 17000
    for name, pol in models.items():
        if pol is None:
            print(f"skip quality {name}: missing ckpt")
            continue
        quality[name] = {}
        for n_dev in (8, 10):
            print(f"=== Commit quality {name} N={n_dev} ===")
            quality[name][n_dev] = commit_quality_rollout(
                pol, protocol, n_devices=n_dev, n=n_eval, seed_base=q_seed + 100 * n_dev, device=device
            )
            q = quality[name][n_dev]
            print(
                f"  exact={q['exact']:.0%} ready_commit={q['ready_commit_rate']:.0%} "
                f"premature|diag={q['premature_among_diag']:.0%} "
                f"premature_ep={q['premature_episode_rate']:.0%}"
            )

    # Reference numbers
    baseline_exact = {6: 1.0, 8: 0.90, 10: 0.42}
    generic_exact = {6: 1.0, 8: 0.86, 10: 0.66}
    gen_json = PROJECT_ROOT / "results" / "p1_commit_ablation.json"
    if gen_json.is_file():
        prev = json.loads(gen_json.read_text(encoding="utf-8"))
        for k, v in (prev.get("generic_exact") or {}).items():
            generic_exact[int(k)] = float(v)

    b_exact = {n: float(b_eval[n]["exact"]) for n in (6, 8, 10)}
    link10_b = _exact_pf(b_eval[10], "link_failure")
    link10_g = 0.92
    if "generic" in quality and 10 in quality["generic"]:
        link10_g = quality["generic"][10]["by_fault"].get("link_failure", {}).get("exact", link10_g)

    # Decision
    if link10_b >= 0.80 and b_exact[10] >= baseline_exact[10] + 0.15:
        decision = "weight_only_sufficient"
        claim = (
            "Diagnose-kind weight alone (no oversample) recovers N=10 Link commit failure — "
            "supports a pure commit-decision objective."
        )
    elif link10_b >= 0.50 and b_exact[10] > baseline_exact[10] + 0.05:
        decision = "weight_helps_oversample_adds"
        claim = (
            "Weight-only helps but undershoots Generic; oversampling still contributes to the full gain."
        )
    else:
        decision = "weight_insufficient_need_oversample"
        claim = "Weight-only does not recover Link; Generic gain likely driven by oversampling / joint heads."

    # Premature check
    prem_note = ""
    if "B_wonly" in quality and "baseline" in quality:
        pb = quality["B_wonly"][10]["premature_among_diag"]
        p0 = quality["baseline"][10]["premature_among_diag"]
        rb = quality["B_wonly"][10]["ready_commit_rate"]
        r0 = quality["baseline"][10]["ready_commit_rate"]
        prem_note = (
            f"N10 ready_commit Baseline {r0:.0%}→B {rb:.0%}; "
            f"premature|diag {p0:.0%}→{pb:.0%}."
        )

    lines = [
        "# Commit factor ablation: weight-only (B) vs Generic",
        "",
        "> **B**: warm-start + w_diag=4, oversample×1, w_hard=1  ",
        "> **Generic**: w_diag=4 + oversample×3 (prior)  ",
        "> Also reports ready-commit vs premature-diagnose rates.",
        f"> Decision: **{decision}**",
        "",
        claim,
        "",
        prem_note,
        "",
        f"- B ckpt: `{CKPT_B.as_posix()}` last_loss={losses[-1]:.4f}",
        "",
        "## Exact",
        "",
        "| N | Baseline | B (weight-only) | Generic |",
        "|---:|---:|---:|---:|",
    ]
    for n in (6, 8, 10):
        lines.append(f"| {n} | {baseline_exact[n]:.0%} | {b_exact[n]:.0%} | {generic_exact[n]:.0%} |")

    lines.extend(
        [
            "",
            "## Per-fault @ N=10 (B)",
            "",
            f"- Device {_pf(b_eval[10], 'device_failure')}",
            f"- Perf {_pf(b_eval[10], 'performance_degradation')}",
            f"- Misconfig {_pf(b_eval[10], 'misconfiguration')}",
            f"- Link {_pf(b_eval[10], 'link_failure')}",
            "",
            "## Ready vs premature (same eval seeds)",
            "",
            "| Model | N | Exact | Ready-commit | Premature\\|diag | Premature ep |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("baseline", "B_wonly", "generic"):
        if name not in quality:
            continue
        for n_dev in (8, 10):
            q = quality[name][n_dev]
            lines.append(
                f"| {name} | {n_dev} | {q['exact']:.0%} | {q['ready_commit_rate']:.0%} | "
                f"{q['premature_among_diag']:.0%} | {q['premature_episode_rate']:.0%} |"
            )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Link@N10 B={link10_b:.0%} (Baseline 0%, Generic ~{link10_g:.0%})",
            f"- N10 Exact B={b_exact[10]:.0%} Generic={generic_exact[10]:.0%} Baseline={baseline_exact[10]:.0%}",
            "",
            "Good intervention signature: ready-commit ↑, premature ≈ flat.",
            "",
            "Run: `python -m src.run_p1_commit_factor_ablation`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "p1_commit_factor_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "decision": decision,
        "claim": claim,
        "losses": losses,
        "b_exact": {str(k): v for k, v in b_exact.items()},
        "b_per_fault": {str(n): b_eval[n].get("per_fault") for n in (6, 8, 10)},
        "quality": quality,
        "baseline_exact": baseline_exact,
        "generic_exact": generic_exact,
    }
    (results / "p1_commit_factor_ablation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(decision)
    print(claim)
    print("Wrote", reports / "p1_commit_factor_ablation.md")


if __name__ == "__main__":
    main()
