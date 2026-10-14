"""Run evaluation suites for baseline or trained policies."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .agents import run_random_valid_episode
from .env_utils import make_eval_env
from .metrics_utils import format_summary_row, summarize_evaluator
from .protocol import PROJECT_ROOT, load_protocol

ActionFn = Callable[[Any], None]


def iter_eval_jobs(protocol: Dict[str, Any]) -> Iterable[Tuple[str, str, str, int, FaultType]]:
    """Yield (group, topology, mesh_label, seed, fault_type)."""
    n = protocol["episodes_per_fault"]
    for ft_name in protocol["fault_types"]:
        fault = FaultType(ft_name)
        base = 100000
        for ep in range(n):
            yield ("seen", "star", "", base + hash(ft_name) % 10000 + ep, fault)
        base = 200000
        for ep in range(n):
            yield ("seen", "linear", "", base + hash(ft_name) % 10000 + ep, fault)
        for label, inst_seed in protocol["mesh_instances"].items():
            base = 300000 + (inst_seed % 1000) * 1000
            for ep in range(n):
                yield ("unseen", "mesh", label, base + hash(ft_name) % 10000 + ep, fault)


def run_suite(action_fn: ActionFn, label: str) -> Dict[str, Any]:
    protocol = load_protocol()
    evaluator = CompetitionEvaluator()
    rows: List[Dict[str, Any]] = []

    for group, topo, mesh_label, seed, fault in iter_eval_jobs(protocol):
        env = make_eval_env([topo], fault_types=[fault], protocol=protocol)
        action_fn(env, seed)
        metrics = env.last_episode_metrics
        if metrics is not None:
            evaluator.add_episode_metrics(metrics)
            rows.append(
                {
                    "agent": label,
                    "group": group,
                    "topology": topo,
                    "mesh_instance": mesh_label,
                    "fault_type": fault.value,
                    "seed": seed,
                    "dsr": int(metrics.diagnosis_success),
                    "steps": metrics.steps,
                    "tool_cost_norm": metrics.tool_cost_normalized,
                    "coverage": metrics.topology_coverage,
                }
            )
        env.close()

    summary = summarize_evaluator(evaluator)
    return {"summary": summary, "rows": rows}


def random_valid_fn(env, seed: int) -> None:
    run_random_valid_episode(env, seed)


def save_results(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-Valid baseline evaluation")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "random_valid_baseline.csv",
    )
    args = parser.parse_args()

    result = run_suite(random_valid_fn, "random_valid")
    save_results(result["rows"], args.out)
    print(format_summary_row("Random-Valid (all)", result["summary"]))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
