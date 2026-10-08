"""GT-free rule ceiling on Star (named baseline, not a learned policy)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_trace import aggregate_behavior
from .diagnosis_decomposition import aggregate_decomposition
from .env_utils import make_masked_eval_env, metrics_env
from .metrics_utils import summarize_evaluator
from .pbrs import find_pbrs
from .protocol import PROJECT_ROOT, load_protocol
from .rule_agent import run_episode_rule
from .run_ablation_nl_poc import _per_fault, _row_consistent
from .run_diagnosis_gate_poc import shortcut_rate


def _evaluate(protocol, *, n: int, seed_base: int) -> Dict:
    cfg = protocol["rule_ceiling_poc"]
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    evaluator = CompetitionEvaluator()
    records: List = []
    summaries: List = []
    rows: List[Dict] = []
    act_by_ft = defaultdict(Counter)

    for i in range(n):
        fault = faults[i % len(faults)]
        env = make_masked_eval_env(
            cfg["topology_types"],
            fault_types=[fault],
            protocol=protocol,
            evidence_obs=True,
            pbrs=True,
            pbrs_apply_shaping=False,
        )
        recs, summary = run_episode_rule(env, episode_id=i, seed=seed_base + i)
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

    timeout = sum(1 for s in summaries if (not s.success) and s.steps >= protocol["max_episode_steps"])
    submitted = sum(1 for r in rows if r["submitted"])
    exact_cons = [r for r in rows if r["success"] and _row_consistent(r)]
    return {
        "stats": summarize_evaluator(evaluator),
        "behavior": aggregate_behavior(summaries, records),
        "decomposition": aggregate_decomposition(summaries),
        "per_fault_dsr": aggregate_behavior(summaries, records).get("fault_type_dsr", {}),
        "per_fault": _per_fault(rows, summaries),
        "shortcut": shortcut_rate(summaries, records),
        "timeout_rate": timeout / max(1, n),
        "submit_rate": submitted / max(1, n),
        "exact_consistent_n": len(exact_cons),
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


def main():
    protocol = load_protocol()
    if "rule_ceiling_poc" not in protocol:
        protocol["rule_ceiling_poc"] = {
            "topology_types": ["star"],
            "eval_episodes": 100,
            "eval_seed_base": 106000,
        }
    cfg = protocol["rule_ceiling_poc"]
    n = int(cfg["eval_episodes"])
    seed_base = int(cfg["eval_seed_base"])
    print(f"Rule ceiling eval n={n} seed_base={seed_base}")
    variant = _evaluate(protocol, n=n, seed_base=seed_base)

    def _pf(ft):
        b = variant["per_fault"].get(ft, {})
        return f"{b.get('dsr', 0):.0%} ({b.get('exact_n', 0)}/{b.get('n', 0)}; cons {b.get('exact_consistent_n', 0)})"

    lines = [
        "# Rule ceiling (GT-free diagnostic tree)",
        "",
        "> No GT in action choice. Uses scan discovery + check_status / check_interfaces / ping results only.",
        "> Star 6-device. Evidence obs ON (for fair env stack with BC); rule reads tool JSON + discovery, not GT.",
        "",
        "| Metric | Rule |",
        "|---|---:|",
        f"| Exact DSR | {variant['decomposition']['exact_diagnosis_accuracy']:.0%} |",
        f"| Type acc | {variant['decomposition']['fault_type_accuracy']:.0%} |",
        f"| Loc acc | {variant['decomposition']['location_accuracy']:.0%} |",
        f"| Shortcut | {variant['shortcut']['shortcut_rate']:.0%} |",
        f"| AvgSteps | {variant['stats']['avg_steps_all']:.1f} |",
        f"| Timeout | {variant['timeout_rate']:.0%} |",
        f"| Device | {_pf('device_failure')} |",
        f"| Perf | {_pf('performance_degradation')} |",
        f"| Misconfig | {_pf('misconfiguration')} |",
        f"| Link | {_pf('link_failure')} |",
        "",
        f"- Misses: **{len(variant['misses'])}** / {n}",
        "",
    ]
    if variant["misses"]:
        lines.append("Sample misses:")
        for m in variant["misses"][:12]:
            lines.append(
                f"- GT {m['fault_type']}@{m['fault_location']} → "
                f"{m['predicted_type']}@{m['predicted_location']} ({m['steps']} steps)"
            )
        lines.append("")

    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "rule_ceiling_poc.md").write_text("\n".join(lines), encoding="utf-8")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("records",)}
        if isinstance(o, list):
            return [_clean(x) for x in o]
        return o

    (results / "rule_ceiling_poc.json").write_text(
        json.dumps({"variant": _clean(variant)}, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        f"exact={variant['decomposition']['exact_diagnosis_accuracy']:.0%} "
        f"steps={variant['stats']['avg_steps_all']:.1f} misses={len(variant['misses'])}"
    )
    print("Wrote", reports / "rule_ceiling_poc.md")


if __name__ == "__main__":
    main()
