"""Metric aggregation helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from netheal.evaluation.metrics import CompetitionEvaluator, EpisodeMetrics


def ttd_success_mean(episodes: List[EpisodeMetrics]) -> Optional[float]:
    success_steps = [m.steps for m in episodes if m.diagnosis_success]
    if not success_steps:
        return None
    return float(np.mean(success_steps))


def summarize_evaluator(evaluator: CompetitionEvaluator) -> Dict[str, Any]:
    summary = evaluator.compute_summary()
    ttd_s = ttd_success_mean(evaluator.episodes)
    return {
        "episodes": len(evaluator.episodes),
        "dsr": summary.get("diagnosis_success_rate", 0.0),
        "macro_f1": summary.get("fault_type_macro_f1", 0.0),
        "avg_steps_all": summary.get("avg_steps", 0.0),
        "normalized_steps_all": summary.get("normalized_steps", 0.0),
        "tool_cost_index": summary.get("tool_cost_index", 0.0),
        "topology_coverage": summary.get("topology_coverage", 0.0),
        "ttd_success": ttd_s,
    }


def format_summary_row(label: str, stats: Dict[str, Any]) -> str:
    ttd = stats.get("ttd_success")
    ttd_str = f"{ttd:.2f}" if ttd is not None else "n/a"
    return (
        f"{label:30s}  DSR={stats['dsr']:.3f}  "
        f"TTD-success={ttd_str}  "
        f"AvgSteps={stats['avg_steps_all']:.2f}  "
        f"Cost={stats['tool_cost_index']:.3f}  "
        f"Cov={stats['topology_coverage']:.3f}  "
        f"n={stats['episodes']}"
    )
