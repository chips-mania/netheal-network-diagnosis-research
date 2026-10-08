"""Step 9.7-A: topology_coverage metric audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from netheal.evaluation.wrapper import MetricsCollectorWrapper
from netheal.faults.injector import FaultType

from .action_mask import wrap_with_action_mask
from .env_utils import make_base_env, make_eval_env, make_masked_eval_env, metrics_env
from .protocol import PROJECT_ROOT, load_protocol


def _coverage_from_env(base_env) -> dict:
    dm = base_env.observation.discovery_matrix
    nodes = len(dm.get_discovered_devices())
    adjacency = np.asarray(dm.adjacency)
    edges = int(np.count_nonzero(np.triu(adjacency)))
    network_size = len(base_env.network.get_all_devices())
    total_edges = len(base_env.network.get_all_connections())
    node_coverage = nodes / max(1, network_size)
    edge_coverage = edges / max(1, total_edges)
    topology_coverage = 0.5 * (node_coverage + edge_coverage)
    return {
        "discovered_nodes": nodes,
        "network_size": network_size,
        "discovered_edges": edges,
        "total_edges": total_edges,
        "node_coverage": node_coverage,
        "edge_coverage": edge_coverage,
        "topology_coverage": topology_coverage,
    }


def _find_scan_action(base_env) -> int:
    from netheal.environment.actions import ActionCategory, TopologyAction

    for aid, spec in base_env.action_space_manager.action_map.items():
        if spec.category == ActionCategory.TOPOLOGY_DISCOVERY and spec.action_type == TopologyAction.SCAN_NETWORK:
            return aid
    raise RuntimeError("scan_network action not found")


def _run_scan_then_diagnose(env, seed: int) -> dict:
    base = env.unwrapped
    env.reset(seed=seed)
    scan_id = _find_scan_action(base)
    env.step(scan_id)
    after_scan = _coverage_from_env(base)

    # Pick first valid diagnosis and end episode (mirrors masked PPO short path)
    diag_id = None
    for aid in base.get_valid_actions():
        spec = base.action_space_manager.get_action_spec(aid)
        if spec and spec.category.value == "diagnosis":
            diag_id = aid
            break
    if diag_id is not None:
        env.step(diag_id)

    metrics = metrics_env(env).last_episode_metrics
    return {
        "after_scan_direct": after_scan,
        "episode_end_metrics": None
        if metrics is None
        else {
            "topology_coverage": metrics.topology_coverage,
            "node_coverage": metrics.node_coverage,
            "edge_coverage": metrics.edge_coverage,
        },
    }


def audit_coverage(protocol: dict, seed: int = 42) -> dict:
    cfg = protocol["ppo_sanity"]
    fault = FaultType.LINK_FAILURE

    unmasked = _run_scan_then_diagnose(
        make_eval_env(cfg["topology_types"], fault_types=[fault], protocol=protocol),
        seed,
    )

    base = make_base_env(cfg["topology_types"], fault_types=[fault], protocol=protocol)
    broken_env = MetricsCollectorWrapper(wrap_with_action_mask(base))
    broken = _run_scan_then_diagnose(broken_env, seed)

    fixed_env = make_masked_eval_env(cfg["topology_types"], fault_types=[fault], protocol=protocol)
    fixed = _run_scan_then_diagnose(fixed_env, seed)

    return {
        "metric_definition": {
            "node_coverage": "discovered_nodes / network_size",
            "edge_coverage": "discovered_edges / total_edges (upper triangle adjacency)",
            "topology_coverage": "0.5 * (node_coverage + edge_coverage)",
            "source": "netheal/evaluation/metrics.py",
            "trace_population": "MetricsCollectorWrapper._finalize_trace reads env.observation.discovery_matrix",
        },
        "seed": seed,
        "unmasked_eval": unmasked,
        "broken_metrics_collector_outside": broken,
        "fixed_action_masker_outside": fixed,
        "root_cause": (
            "When MetricsCollectorWrapper wraps ActionMasker, self.env has no .observation; "
            "discovered_nodes/edges remain 0 at finalize → topology_coverage=0 even after scan."
        ),
        "fix": "ActionMasker(MetricsCollectorWrapper(base)) — see make_masked_eval_env().",
    }


def build_report(audit: dict) -> str:
    def fmt_ep(label: str, block: dict) -> str:
        scan = block["after_scan_direct"]
        end = block["episode_end_metrics"]
        end_cov = end["topology_coverage"] if end else 0.0
        return (
            f"| {label} | {scan['topology_coverage']:.3f} (direct after scan) | "
            f"{end_cov:.3f} (metrics at episode end) | {scan['discovered_nodes']}/{scan['network_size']} nodes |"
        )

    lines = [
        "# Topology Coverage Metric Audit (Step 9.7-A)",
        "",
        "## Official definition",
        "",
        "```",
        "node_coverage     = discovered_nodes / network_size",
        "edge_coverage     = discovered_edges / total_edges",
        "topology_coverage = 0.5 * (node_coverage + edge_coverage)",
        "```",
        "",
        "Populated from `observation.discovery_matrix` at **episode end** in `MetricsCollectorWrapper._finalize_trace`.",
        "",
        "## Root cause of Coverage=0 in Step 9.6 masked eval",
        "",
        audit["root_cause"],
        "",
        f"**Fix:** {audit['fix']}",
        "",
        "## Empirical comparison (scan → diagnose, seed=42)",
        "",
        "| Wrapper order | After scan (direct) | Episode-end metrics | Nodes |",
        "|---|---:|---:|---|",
        fmt_ep("Unmasked MetricsCollector(base)", audit["unmasked_eval"]),
        fmt_ep("Broken: MetricsCollector(ActionMasker)", audit["broken_metrics_collector_outside"]),
        fmt_ep("Fixed: ActionMasker(MetricsCollector)", audit["fixed_action_masker_outside"]),
        "",
        "## Takeaway",
        "",
        "- `scan_network` **does** increase discovery → direct coverage >0 after scan.",
        "- Broken wrapper order zeroes out metrics at episode end.",
        "- Step 9.7+ uses fixed wrapper; re-evaluate Coverage before Mesh generalization claims.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    protocol = load_protocol()
    audit = audit_coverage(protocol)
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "topology_coverage_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    path = reports / "topology_coverage_audit.md"
    path.write_text(build_report(audit), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
