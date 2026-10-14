"""Diagnose commit margin audit: P(diagnose), rank, margin on live P1 states.

Splits: all steps / conclude-ready / timeout episodes / last K steps of timeouts.
Compares Star N=6/8/10 to separate calibration vs near-zero commit mass.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from netheal.environment.actions import ActionCategory
from netheal.faults.injector import FaultType

from .action_trace import _base_env, _format_target, _is_action_valid, _parse_action_spec
from .env_utils import make_masked_eval_env, metrics_env
from .graph_obs import build_graph_obs
from .graph_policy import KIND_ORDER, GraphPointerPolicy
from .pointer_actions import resolve_pointer
from .protocol import PROJECT_ROOT, load_protocol
from .rule_agent_v2 import RuleAgentV2
from .run_p1_star10_failure_audit import CKPT, ensure_p1, evidence_flags, policy_logits
from .run_symmetric_reward_poc import set_seeds
from .star_pad import steps_budget

DIAG_I = next(i for i, k in enumerate(KIND_ORDER) if k.value == "diagnose")


def kind_stats(logits) -> Dict[str, float]:
    probs = F.softmax(logits["kind"][0], dim=-1).detach().cpu().numpy()
    scores = logits["kind"][0].detach().cpu().numpy()
    top = int(np.argmax(probs))
    # 1-indexed rank of diagnose (1 = best)
    rank = float(1 + np.sum(scores > scores[DIAG_I]))
    # margin: logit(diagnose) - logit(top non-diagnose) ; or vs argmax if diagnose is top
    others = [scores[i] for i in range(len(scores)) if i != DIAG_I]
    best_other = max(others) if others else scores[DIAG_I]
    margin = float(scores[DIAG_I] - best_other)
    return {
        "p_diagnose": float(probs[DIAG_I]),
        "p_top": float(probs[top]),
        "top_kind": KIND_ORDER[top].value,
        "diag_rank": rank,
        "diag_margin": margin,
        "diag_is_top": float(top == DIAG_I),
        "diag_in_top2": float(rank <= 2),
        "diag_in_top3": float(rank <= 3),
    }


def _summarize(rows: List[Dict]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    pd = [r["p_diagnose"] for r in rows]
    mg = [r["diag_margin"] for r in rows]
    rk = [r["diag_rank"] for r in rows]
    top_kinds = defaultdict(int)
    for r in rows:
        top_kinds[r["top_kind"]] += 1
    return {
        "n": len(rows),
        "p_diagnose_mean": float(np.mean(pd)),
        "p_diagnose_median": float(np.median(pd)),
        "p_diagnose_p10": float(np.percentile(pd, 10)),
        "p_diagnose_p90": float(np.percentile(pd, 90)),
        "margin_mean": float(np.mean(mg)),
        "margin_median": float(np.median(mg)),
        "rank_mean": float(np.mean(rk)),
        "diag_is_top": float(np.mean([r["diag_is_top"] for r in rows])),
        "diag_in_top2": float(np.mean([r["diag_in_top2"] for r in rows])),
        "diag_in_top3": float(np.mean([r["diag_in_top3"] for r in rows])),
        "top_kind_hist": dict(top_kinds),
    }


def rollout_margin(
    policy: GraphPointerPolicy,
    protocol,
    *,
    n_devices: int,
    n: int,
    seed_base: int,
    device: torch.device,
    last_k: int = 3,
) -> Dict:
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    steps_max = steps_budget(n_devices, floor=int(protocol["max_episode_steps"]))
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    by_ft_ready: Dict[str, List[Dict]] = defaultdict(list)
    ep_rows: List[Dict] = []

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
        fault_type = gt.get("type", "")
        fault_location = gt.get("location", "")
        agent_ev.reset()
        step_stats: List[Dict] = []
        ready_step = None
        saw_dx = False
        step = 0

        while True:
            step += 1
            # flags from *current* mem before action (post previous tools)
            flags = evidence_flags(agent_ev.mem, fault_type, fault_location)
            g = build_graph_obs(env, mem=agent_ev.mem)
            logits = policy_logits(policy, g, device)
            st = kind_stats(logits)
            st.update(
                {
                    "step": step,
                    "fault_type": fault_type,
                    "conclude_match": bool(flags.get("conclude_match")),
                    "conclude_ready": bool(flags.get("conclude_ready")),
                }
            )
            step_stats.append(st)
            buckets["all"].append(st)
            if st["conclude_match"]:
                buckets["conclude_match"].append(st)
                by_ft_ready[fault_type].append(st)
                if ready_step is None:
                    ready_step = step
                    buckets["first_conclude_match"].append(st)

            ptr = policy.act(g, device)
            action_id = int(resolve_pointer(env, ptr))
            obs, reward, terminated, truncated, info = env.step(action_id)
            agent_ev.observe_step(info)
            agent_ev.refresh_plan(env)

            spec = _parse_action_spec(info, base, action_id)
            phase = info.get("factorized_phase")
            if spec and spec.category == ActionCategory.DIAGNOSIS and phase != "type_commit":
                saw_dx = True

            if terminated or truncated:
                break

        m = metrics_env(env).last_episode_metrics
        success = bool(m and m.diagnosis_success)
        timeout = (not success) and step >= steps_max
        # last-K of timeout eps
        if timeout:
            for st in step_stats[-last_k:]:
                buckets["timeout_last_k"].append(st)
            for st in step_stats:
                if st["conclude_match"]:
                    buckets["timeout_conclude_match"].append(st)
            if step_stats:
                buckets["timeout_final_step"].append(step_stats[-1])

        # after success, last step should often be diagnose
        if success and step_stats:
            buckets["success_final_step"].append(step_stats[-1])

        ep_rows.append(
            {
                "fault_type": fault_type,
                "success": success,
                "timeout": timeout,
                "submitted": saw_dx,
                "ready_step": ready_step,
                "steps": step,
                "final_p_diag": step_stats[-1]["p_diagnose"] if step_stats else None,
                "final_margin": step_stats[-1]["diag_margin"] if step_stats else None,
                "final_rank": step_stats[-1]["diag_rank"] if step_stats else None,
                "mean_p_diag_when_ready": (
                    float(np.mean([s["p_diagnose"] for s in step_stats if s["conclude_match"]]))
                    if any(s["conclude_match"] for s in step_stats)
                    else None
                ),
            }
        )
        env.close()

    # per-fault summaries on conclude_match steps
    by_ft_sum = {ft: _summarize(rows) for ft, rows in sorted(by_ft_ready.items())}

    # episode-level: among timeout+ever-ready, final step stats
    to_ready = [
        e
        for e in ep_rows
        if e["timeout"] and e["mean_p_diag_when_ready"] is not None
    ]
    return {
        "n_devices": n_devices,
        "n": n,
        "max_steps": steps_max,
        "buckets": {k: _summarize(v) for k, v in buckets.items()},
        "by_fault_conclude_match": by_ft_sum,
        "episodes": {
            "exact": sum(1 for e in ep_rows if e["success"]) / max(1, len(ep_rows)),
            "timeout": sum(1 for e in ep_rows if e["timeout"]) / max(1, len(ep_rows)),
            "timeout_with_ready": len(to_ready) / max(1, len(ep_rows)),
            "timeout_ready_final_p_diag_mean": (
                float(np.mean([e["final_p_diag"] for e in to_ready])) if to_ready else None
            ),
            "timeout_ready_final_margin_mean": (
                float(np.mean([e["final_margin"] for e in to_ready])) if to_ready else None
            ),
            "timeout_ready_final_rank_mean": (
                float(np.mean([e["final_rank"] for e in to_ready])) if to_ready else None
            ),
        },
    }


def _pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:.0%}"


def _f(x, nd=3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def main() -> None:
    protocol = load_protocol()
    cfg = protocol.get("graph_expert_bc_poc", {})
    set_seeds(int(cfg.get("policy_seed", 1)))
    device = torch.device("cpu")
    policy = ensure_p1(protocol, device)
    policy.eval()

    n_eval = int(cfg.get("eval_episodes", 100))
    seed = int(cfg.get("eval_seed_base", 130000)) + 5000  # offset from prior audit

    results = {}
    for n_dev in (6, 8, 10):
        print(f"=== Commit margin Star N={n_dev} ===")
        results[n_dev] = rollout_margin(
            policy,
            protocol,
            n_devices=n_dev,
            n=n_eval,
            seed_base=seed + 100 * n_dev,
            device=device,
            last_k=3,
        )
        b = results[n_dev]["buckets"]
        cm = b.get("conclude_match", {})
        tf = b.get("timeout_final_step", {})
        print(
            f"  conclude_match: n={cm.get('n', 0)} P(diag)={_f(cm.get('p_diagnose_mean'))} "
            f"margin={_f(cm.get('margin_mean'))} rank={_f(cm.get('rank_mean'), 2)} "
            f"top1={_pct(cm.get('diag_is_top'))} top2={_pct(cm.get('diag_in_top2'))}"
        )
        print(
            f"  timeout_final: n={tf.get('n', 0)} P(diag)={_f(tf.get('p_diagnose_mean'))} "
            f"margin={_f(tf.get('margin_mean'))} top_kinds={tf.get('top_kind_hist')}"
        )

    # Narrative
    cm6 = results[6]["buckets"].get("conclude_match", {})
    cm10 = results[10]["buckets"].get("conclude_match", {})
    tf10 = results[10]["buckets"].get("timeout_final_step", {})
    tcm10 = results[10]["buckets"].get("timeout_conclude_match", {})

    p6 = cm6.get("p_diagnose_mean") or 0
    p10 = cm10.get("p_diagnose_mean") or 0
    top2_10 = cm10.get("diag_in_top2") or 0
    p_tf = tf10.get("p_diagnose_mean") or 0

    if p10 >= 0.20 and top2_10 >= 0.40:
        narrative = (
            "On conclude-ready states at N=10, diagnose often has **non-trivial mass / top-2 rank** "
            "→ closer to **decision-boundary / calibration** than zero commit."
        )
    elif p_tf < 0.05 and (tf10.get("diag_is_top") or 0) < 0.05:
        narrative = (
            "On N=10 timeout final steps, P(diagnose) is **near-zero and almost never top** "
            "→ commit concept **does not transfer**; not a thin margin issue."
        )
    else:
        narrative = (
            "Mixed: inspect conclude-match vs timeout-final rows — may be boundary on ready states "
            "but collapsed commit after prolonged tool loops."
        )

    lines = [
        "# P1 diagnose commit margin audit",
        "",
        f"> Checkpoint: `{CKPT.as_posix()}`",
        "> Question: on evidence-ready / timeout states, is diagnose 2nd–3rd (calibration) or ~0 (no commit)?",
        f"> Read: {narrative}",
        "",
        "## Conclude-match steps (Rule `_try_conclude` == GT from same mem)",
        "",
        "| N | n | P(diag) mean | median | margin mean | rank mean | diag top-1 | top-2 | top-3 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n_dev in (6, 8, 10):
        s = results[n_dev]["buckets"].get("conclude_match", {"n": 0})
        lines.append(
            f"| {n_dev} | {s.get('n', 0)} | {_f(s.get('p_diagnose_mean'))} | {_f(s.get('p_diagnose_median'))} | "
            f"{_f(s.get('margin_mean'))} | {_f(s.get('rank_mean'), 2)} | {_pct(s.get('diag_is_top'))} | "
            f"{_pct(s.get('diag_in_top2'))} | {_pct(s.get('diag_in_top3'))} |"
        )

    lines.extend(
        [
            "",
            "## Timeout final step",
            "",
            "| N | n | P(diag) | margin | rank | diag top-1 | top-2 | dominant top kinds |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for n_dev in (6, 8, 10):
        s = results[n_dev]["buckets"].get("timeout_final_step", {"n": 0})
        hist = s.get("top_kind_hist") or {}
        top3 = sorted(hist.items(), key=lambda kv: -kv[1])[:3]
        hist_s = ", ".join(f"{k}:{c}" for k, c in top3) if top3 else "—"
        lines.append(
            f"| {n_dev} | {s.get('n', 0)} | {_f(s.get('p_diagnose_mean'))} | {_f(s.get('margin_mean'))} | "
            f"{_f(s.get('rank_mean'), 2)} | {_pct(s.get('diag_is_top'))} | {_pct(s.get('diag_in_top2'))} | {hist_s} |"
        )

    lines.extend(
        [
            "",
            "## Timeout ∩ conclude-match steps (smoking gun bucket)",
            "",
            "| N | n | P(diag) | margin | rank | top-1 | top-2 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for n_dev in (6, 8, 10):
        s = results[n_dev]["buckets"].get("timeout_conclude_match", {"n": 0})
        lines.append(
            f"| {n_dev} | {s.get('n', 0)} | {_f(s.get('p_diagnose_mean'))} | {_f(s.get('margin_mean'))} | "
            f"{_f(s.get('rank_mean'), 2)} | {_pct(s.get('diag_is_top'))} | {_pct(s.get('diag_in_top2'))} |"
        )

    lines.extend(
        [
            "",
            "## Success final step (sanity: should be diagnose)",
            "",
            "| N | n | P(diag) | diag top-1 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for n_dev in (6, 8, 10):
        s = results[n_dev]["buckets"].get("success_final_step", {"n": 0})
        lines.append(
            f"| {n_dev} | {s.get('n', 0)} | {_f(s.get('p_diagnose_mean'))} | {_pct(s.get('diag_is_top'))} |"
        )

    lines.extend(
        [
            "",
            "## Per-fault conclude-match @ N=10",
            "",
            "| Fault | n | P(diag) | margin | top-1 | top-2 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for ft, s in results[10]["by_fault_conclude_match"].items():
        lines.append(
            f"| {ft} | {s.get('n', 0)} | {_f(s.get('p_diagnose_mean'))} | {_f(s.get('margin_mean'))} | "
            f"{_pct(s.get('diag_is_top'))} | {_pct(s.get('diag_in_top2'))} |"
        )

    # Compare N6 vs N10 deltas
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Conclude-match P(diag) N6→N10: **{_f(p6)} → {_f(p10)}** (Δ {p10 - p6:+.3f})",
            f"- Conclude-match diagnose top-2 @N10: **{_pct(top2_10)}**",
            f"- Timeout final P(diag) @N10: **{_f(p_tf)}**; top-1 diagnose **{_pct(tf10.get('diag_is_top'))}**",
            f"- Timeout∩ready P(diag) @N10: **{_f(tcm10.get('p_diagnose_mean'))}**; "
            f"top-2 **{_pct(tcm10.get('diag_in_top2'))}**",
            "",
            narrative,
            "",
            "If near-zero on timeout-final → strengthen diagnose when evidence-ready (loss/curriculum), not just temperature.",
            "If top-2 high but top-1 low → calibration / margin sharpening may suffice.",
            "",
            "Run: `python -m src.run_p1_commit_margin_audit`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results_dir = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (reports / "p1_commit_margin_audit.md").write_text("\n".join(lines), encoding="utf-8")
    (results_dir / "p1_commit_margin_audit.json").write_text(
        json.dumps({"narrative": narrative, "by_n": {str(k): v for k, v in results.items()}}, indent=2, default=str),
        encoding="utf-8",
    )
    print(narrative)
    print("Wrote", reports / "p1_commit_margin_audit.md")


if __name__ == "__main__":
    main()
