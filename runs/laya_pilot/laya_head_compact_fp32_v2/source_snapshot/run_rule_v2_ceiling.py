"""Rule v2 ceiling across Star / Linear / Mesh (Phase 1 of GNN_expert_guided_spec)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_trace import aggregate_behavior
from .diagnosis_decomposition import aggregate_decomposition
from .env_utils import make_masked_eval_env, metrics_env
from .metrics_utils import summarize_evaluator
from .pbrs import find_pbrs
from .protocol import PROJECT_ROOT, load_protocol
from .rule_agent_v2 import run_episode_rule_v2
from .run_ablation_nl_poc import _per_fault, _row_consistent
from .run_diagnosis_gate_poc import shortcut_rate
from .star_pad import steps_budget


def _evaluate(
    protocol,
    *,
    topology: str,
    n_devices: int,
    n: int,
    seed_base: int,
    max_episode_steps: Optional[int] = None,
) -> Dict:
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    evaluator = CompetitionEvaluator()
    records: List = []
    summaries: List = []
    rows: List[Dict] = []
    act_by_ft = defaultdict(Counter)
    steps = int(max_episode_steps or steps_budget(n_devices, floor=int(protocol["max_episode_steps"])))

    for i in range(n):
        fault = faults[i % len(faults)]
        env = make_masked_eval_env(
            [topology],
            fault_types=[fault],
            protocol=protocol,
            evidence_obs=True,
            pbrs=True,
            pbrs_apply_shaping=False,
            min_devices=n_devices,
            max_devices=n_devices,
            max_episode_steps=steps,
        )
        recs, summary = run_episode_rule_v2(env, episode_id=i, seed=seed_base + i)
        m = metrics_env(env).last_episode_metrics
        if m:
            evaluator.add_episode_metrics(m)
        tracker = find_pbrs(env)
        bits = tracker.bits_before_diagnosis or tracker.bits.snapshot()
        phi = tracker.phi_before_diagnosis
        if phi is None:
            phi = tracker.bits.phi
        rows.append(
            {
                "fault_type": summary.fault_type,
                "fault_location": summary.fault_location,
                "success": summary.success,
                "predicted_type": summary.predicted_type,
                "predicted_location": summary.predicted_location,
                "phi_before_dx": int(phi),
                "type_bit": int(bits.get("type_bit", 0)),
                "loc_bit": int(bits.get("loc_bit", 0)),
                "steps": summary.steps,
                "submitted": summary.predicted_type is not None,
            }
        )
        for r in recs:
            if r.action_category in ("diagnostic", "topology_discovery", "diagnosis"):
                act_by_ft[summary.fault_type][r.action_type] += 1
        records.extend(recs)
        summaries.append(summary)
        env.close()

    timeout = sum(1 for s in summaries if (not s.success) and s.steps >= steps)
    submitted = sum(1 for r in rows if r["submitted"])
    exact_cons = [r for r in rows if r["success"] and _row_consistent(r)]
    exact = float(aggregate_decomposition(summaries)["exact_diagnosis_accuracy"])
    return {
        "topology": topology,
        "n_devices": n_devices,
        "max_episode_steps": steps,
        "stats": summarize_evaluator(evaluator),
        "behavior": aggregate_behavior(summaries, records),
        "decomposition": aggregate_decomposition(summaries),
        "per_fault": _per_fault(rows, summaries),
        "shortcut": shortcut_rate(summaries, records),
        "timeout_rate": timeout / max(1, n),
        "submit_rate": submitted / max(1, n),
        "exact_consistent_n": len(exact_cons),
        "exact": exact,
        "action_types_by_fault": {k: dict(v) for k, v in act_by_ft.items()},
        "n": n,
        "misses": [
            {
                "fault_type": r["fault_type"],
                "fault_location": r["fault_location"],
                "predicted_type": r["predicted_type"],
                "predicted_location": r["predicted_location"],
                "steps": r["steps"],
            }
            for r in rows
            if not r["success"]
        ],
    }


def _pf(variant: Dict, ft: str) -> str:
    b = variant["per_fault"].get(ft, {})
    return (
        f"{b.get('dsr', 0):.0%} ({b.get('exact_n', 0)}/{b.get('n', 0)}; "
        f"cons {b.get('exact_consistent_n', 0)})"
    )


def _gate(results: Dict) -> Dict:
    star6 = results.get("star_6", {})
    star8 = results.get("star_8", {})
    star10 = results.get("star_10", {})
    mesh = results.get("mesh_6", {})
    e6 = float(star6.get("exact", 0))
    e8 = float(star8.get("exact", 0))
    e10 = float(star10.get("exact", 0))
    em = float(mesh.get("exact", 0))
    star_ok = e6 >= 0.95 and e8 >= 0.90 and e10 >= 0.90
    mesh_ok = em >= 0.70
    mesh_doc_only = em < 0.70
    policy_block = em < 0.50
    if star_ok and mesh_ok:
        decision = "go"
    elif star_ok and mesh_doc_only and not policy_block:
        decision = "go_with_mesh_caveat"
    elif policy_block:
        decision = "no_go_fix_expert"
    else:
        decision = "no_go_star"
    return {
        "decision": decision,
        "star_ok": star_ok,
        "mesh_ok": mesh_ok,
        "policy_learning_blocked": policy_block,
        "thresholds": {"star6": 0.95, "star8_10": 0.90, "mesh": 0.70, "block": 0.50},
        "exact": {"star_6": e6, "star_8": e8, "star_10": e10, "mesh_6": em},
    }


def main() -> None:
    protocol = load_protocol()
    cfg = protocol.setdefault(
        "rule_v2_ceiling_poc",
        {
            "eval_episodes": 100,
            "mesh_episodes": 100,
            "eval_seed_base": 120000,
            "mesh_max_steps": 40,
        },
    )
    n = int(cfg["eval_episodes"])
    n_mesh = int(cfg.get("mesh_episodes", n))
    seed = int(cfg["eval_seed_base"])
    results: Dict = {}

    suites = [
        ("star_6", "star", 6, n, seed, None),
        ("star_8", "star", 8, n, seed + 1000, None),
        ("star_10", "star", 10, n, seed + 2000, None),
        ("linear_6", "linear", 6, n, seed + 3000, None),
        ("mesh_6", "mesh", 6, n_mesh, seed + 4000, int(cfg.get("mesh_max_steps", 40))),
    ]

    for key, topo, nd, ne, sb, mx in suites:
        print(f"=== Rule v2 ceiling {key} ===")
        results[key] = _evaluate(
            protocol, topology=topo, n_devices=nd, n=ne, seed_base=sb, max_episode_steps=mx
        )
        v = results[key]
        print(
            f"  exact={v['exact']:.0%} timeout={v['timeout_rate']:.0%} "
            f"steps={v['stats']['avg_steps_all']:.1f} misses={len(v['misses'])}"
        )

    gate = _gate(results)

    lines = [
        "# Rule v2 ceiling (topology-agnostic)",
        "",
        "> GT-free. No Star `center` / spoke expectations.",
        "> Nominal edges from network graph skeleton vs post-scan discovery CONNECTED.",
        "> Phase 1 of `GNN_expert_guided_spec.md`.",
        "",
        f"- Decision: **{gate['decision']}**",
        f"- Star thresholds met: {gate['star_ok']}; Mesh≥70%: {gate['mesh_ok']}; "
        f"policy blocked (Mesh&lt;50%): {gate['policy_learning_blocked']}",
        "",
        "## Exact by setting",
        "",
        "| Setting | Exact | Shortcut | Timeout | AvgSteps | Misses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, *_ in suites:
        v = results[key]
        lines.append(
            f"| {key} | {v['exact']:.0%} | {v['shortcut']['shortcut_rate']:.0%} | "
            f"{v['timeout_rate']:.0%} | {v['stats']['avg_steps_all']:.1f} | {len(v['misses'])} |"
        )

    lines.extend(["", "## Per-fault (selected)", ""])
    for key in ("star_6", "star_8", "star_10", "linear_6", "mesh_6"):
        v = results[key]
        lines.append(f"### {key}")
        lines.append(f"- Device {_pf(v, 'device_failure')}")
        lines.append(f"- Perf {_pf(v, 'performance_degradation')}")
        lines.append(f"- Misconfig {_pf(v, 'misconfiguration')}")
        lines.append(f"- Link {_pf(v, 'link_failure')}")
        if v["misses"]:
            lines.append(f"- Sample misses ({min(8, len(v['misses']))}/{len(v['misses'])}):")
            for m in v["misses"][:8]:
                lines.append(
                    f"  - GT {m['fault_type']}@{m['fault_location']} → "
                    f"{m['predicted_type']}@{m['predicted_location']} ({m['steps']} steps)"
                )
        lines.append("")

    lines.extend(
        [
            "## Go / No-Go (spec)",
            "",
            "- Star N=6 ≥95%, N=8/10 ≥90%: required for Graph BC Phase 3",
            "- Mesh ≥70%: full go; 50–70%: go_with_mesh_caveat (document failure modes); "
            "&lt;50%: **no_go_fix_expert** (do not train Graph policy yet)",
            "",
            "Run: `python -m src.run_rule_v2_ceiling`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results_dir = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (reports / "rule_v2_ceiling.md").write_text("\n".join(lines), encoding="utf-8")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("records",)}
        if isinstance(o, list):
            return [_clean(x) for x in o]
        return o

    (results_dir / "rule_v2_ceiling.json").write_text(
        json.dumps({"gate": gate, "results": _clean(results)}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"decision={gate['decision']} exacts={gate['exact']}")
    print("Wrote", reports / "rule_v2_ceiling.md")


if __name__ == "__main__":
    main()
