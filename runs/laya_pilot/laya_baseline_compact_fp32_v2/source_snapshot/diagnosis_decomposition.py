"""Decompose exact diagnosis (DSR) into fault-type and location accuracy."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .action_trace import EpisodeSummary


def _normalize_edge_location(location: str) -> Tuple[str, ...]:
    if not location or "->" not in location:
        return (location.strip(),) if location else tuple()
    return tuple(sorted(p.strip() for p in location.split("->") if p.strip()))


def fault_type_match(actual: str, predicted: Optional[str]) -> bool:
    if not predicted:
        return False
    return actual == predicted


def location_match(actual: str, predicted: Optional[str], fault_type: str) -> bool:
    if not predicted:
        return False
    if fault_type in ("link_failure", "performance_degradation"):
        return _normalize_edge_location(actual) == _normalize_edge_location(predicted)
    actual_norm = actual.strip()
    pred_norm = predicted.strip()
    if actual_norm == pred_norm:
        return True
    # device/misconfig: predicted edge link form may still reference same device
    if "->" in pred_norm:
        parts = [p.strip() for p in pred_norm.split("->") if p.strip()]
        return actual_norm in parts
    return False


def decompose_episode(summary: EpisodeSummary) -> Dict[str, bool]:
    type_ok = fault_type_match(summary.fault_type, summary.predicted_type)
    loc_ok = location_match(summary.fault_location, summary.predicted_location, summary.fault_type)
    exact_ok = summary.success
    return {
        "episode_id": summary.episode_id,
        "actual_type": summary.fault_type,
        "actual_location": summary.fault_location,
        "predicted_type": summary.predicted_type,
        "predicted_location": summary.predicted_location,
        "fault_type_correct": type_ok,
        "location_correct": loc_ok,
        "exact_correct": exact_ok,
    }


def aggregate_decomposition(summaries: List[EpisodeSummary]) -> Dict:
    rows = [decompose_episode(s) for s in summaries]
    n = max(1, len(rows))
    return {
        "episodes": len(rows),
        "fault_type_accuracy": sum(r["fault_type_correct"] for r in rows) / n,
        "location_accuracy": sum(r["location_correct"] for r in rows) / n,
        "exact_diagnosis_accuracy": sum(r["exact_correct"] for r in rows) / n,
        "rows": rows,
    }


def format_matching_table(rows: List[Dict], limit: int = 20) -> List[str]:
    lines = [
        "| Ep | Actual loc | Pred type | Pred loc | Type OK | Loc OK | Exact |",
        "|---:|---|---|---|:---:|:---:|:---:|",
    ]
    for r in rows[:limit]:
        lines.append(
            f"| {r['episode_id']} | `{r['actual_location']}` | "
            f"{r['predicted_type'] or '-'} | `{r['predicted_location'] or '-'}` | "
            f"{'Y' if r['fault_type_correct'] else 'N'} | "
            f"{'Y' if r['location_correct'] else 'N'} | "
            f"{'Y' if r['exact_correct'] else 'N'} |"
        )
    if len(rows) > limit:
        lines.append(f"| … | ({len(rows) - limit} more episodes) | | | | | |")
    return lines
