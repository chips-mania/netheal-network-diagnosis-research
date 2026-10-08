"""Step 9.8 — Post-scan information audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from netheal.environment.actions import ActionCategory, TopologyAction
from netheal.faults.injector import FaultType

from .controlled_faults import anchor_description, inject_controlled_fault
from .env_utils import make_base_env
from .obs_features import (
    FIELD_NAMES,
    diff_summary,
    flatten_obs,
    full_vector_distance,
    pairwise_obs_distance,
    summarize_obs,
)
from .protocol import PROJECT_ROOT, load_protocol


def _find_scan_action(env) -> int:
    for aid, spec in env.action_space_manager.action_map.items():
        if spec.category == ActionCategory.TOPOLOGY_DISCOVERY and spec.action_type == TopologyAction.SCAN_NETWORK:
            return aid
    raise RuntimeError("scan_network not found")


def collect_reset_and_post_scan(
    env,
    *,
    seed: int,
    fault_type: FaultType,
) -> Dict[str, Any]:
    env.reset(seed=seed)
    fault = inject_controlled_fault(env, fault_type)
    obs_reset, _ = env.observation.to_dict(), None
    obs_reset = env.observation.to_dict()

    scan_id = _find_scan_action(env)
    env.step(scan_id)
    obs_post = env.observation.to_dict()

    return {
        "seed": seed,
        "fault_type": fault_type.value,
        "fault_location": fault.location,
        "obs_reset_summary": summarize_obs(obs_reset),
        "obs_post_scan_summary": summarize_obs(obs_post),
        "obs_diff": diff_summary(obs_reset, obs_post),
        "pairwise_reset_vs_post": pairwise_obs_distance(obs_reset, obs_post),
        "post_scan_vector": flatten_obs(obs_post).tolist(),
        "delta_vector": (flatten_obs(obs_post) - flatten_obs(obs_reset)).tolist(),
    }


def controlled_four_fault_snapshot(network_seed: int = 42) -> Dict[str, Any]:
    protocol = load_protocol()
    env = make_base_env(["star"], protocol=protocol)
    samples = []
    for ft in protocol["fault_types"]:
        samples.append(collect_reset_and_post_scan(env, seed=network_seed, fault_type=FaultType(ft)))

    # Pairwise distances between post-scan observations (same topology, different faults)
    distances = {}
    for i, a in enumerate(samples):
        for j, b in enumerate(samples):
            if j <= i:
                continue
            key = f"{a['fault_type']}__vs__{b['fault_type']}"
            va = np.array(a["post_scan_vector"], dtype=np.float32)
            vb = np.array(b["post_scan_vector"], dtype=np.float32)
            distances[key] = float(np.linalg.norm(va - vb))

    env.close()
    return {
        "network_seed": network_seed,
        "anchor": anchor_description(),
        "samples": samples,
        "post_scan_pairwise_l2": distances,
    }


def build_probe_dataset(
    n_seeds: int = 200,
    seed_start: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    protocol = load_protocol()
    fault_types = [FaultType(ft) for ft in protocol["fault_types"]]
    env = make_base_env(["star"], protocol=protocol)

    xs: List[np.ndarray] = []
    ys: List[int] = []
    seeds_used: List[int] = []

    for i in range(n_seeds):
        seed = seed_start + i
        fault = fault_types[i % len(fault_types)]
        record = collect_reset_and_post_scan(env, seed=seed, fault_type=fault)
        xs.append(np.array(record["post_scan_vector"], dtype=np.float32))
        ys.append(fault_types.index(fault))
        seeds_used.append(seed)

    env.close()
    return np.stack(xs), np.array(ys, dtype=np.int64), seeds_used


def run_supervised_probe(X: np.ndarray, y: np.ndarray, n_classes: int = 4) -> Dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    if len(X) < n_classes * 2:
        return {"error": "not enough samples"}

    cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y))), shuffle=True, random_state=0)
    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=0)),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=0),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=0
        ),
    }

    random_baseline = 1.0 / n_classes
    results = {"random_chance": random_baseline, "models": {}}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        results["models"][name] = {
            "cv_accuracy_mean": float(np.mean(scores)),
            "cv_accuracy_std": float(np.std(scores)),
            "cv_scores": [float(s) for s in scores],
        }

    # Per-field probes (discovery only, etc.)
    field_results = {}
    dim_cursor = 0
    protocol = load_protocol()
    env = make_base_env(["star"], protocol=protocol)
    env.reset(seed=1000)
    obs0 = env.observation.to_dict()
    env.close()
    field_dims = {f: int(np.asarray(obs0[f]).size) for f in FIELD_NAMES}

    for field, dim in field_dims.items():
        sl = slice(dim_cursor, dim_cursor + dim)
        dim_cursor += dim
        X_field = X[:, sl]
        if np.allclose(X_field.std(axis=0), 0):
            field_results[field] = {"cv_accuracy_mean": None, "note": "zero variance"}
            continue
        lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
        scores = cross_val_score(lr, X_field, y, cv=cv, scoring="accuracy")
        field_results[field] = {
            "cv_accuracy_mean": float(np.mean(scores)),
            "cv_accuracy_std": float(np.std(scores)),
        }

    results["per_field_logistic"] = field_results
    return results


def interpret_probe(probe: Dict[str, Any], pairwise: Dict[str, float]) -> Dict[str, Any]:
    random_chance = probe.get("random_chance", 0.25)
    best = 0.0
    best_model = None
    for name, stats in probe.get("models", {}).items():
        acc = stats.get("cv_accuracy_mean", 0.0)
        if acc > best:
            best = acc
            best_model = name

    nonzero_dist = [v for v in pairwise.values() if v > 1e-6]
    faults_separable_in_obs = len(nonzero_dist) > 0

    if best >= 0.75:
        branch = "A"
        recommendation = (
            "Scan-only post-observation carries strong fault-type signal. "
            "Reframe study toward diagnosis-mapping generalization; TTD/Tool Cost secondary."
        )
    elif best >= 0.45:
        branch = "B_partial"
        recommendation = (
            "Some scan-only separability but not decisive. Multi-fault masked training may help; "
            "also audit link-only shortcut vs other faults."
        )
    else:
        branch = "B"
        recommendation = (
            "Scan-only fault-type separability near chance. Multi-step diagnostics likely needed; "
            "proceed with masked multi-fault training experiments."
        )

    return {
        "best_model": best_model,
        "best_cv_accuracy": best,
        "random_chance": random_chance,
        "faults_differ_post_scan": faults_separable_in_obs,
        "branch": branch,
        "recommendation": recommendation,
    }


def build_markdown(
    snapshot: Dict[str, Any],
    probe: Dict[str, Any],
    interpretation: Dict[str, Any],
    n_probe_seeds: int,
) -> str:
    lines = [
        "# Step 9.8 — Post-Scan Information Audit",
        "",
        "## Setup",
        "",
        f"- Topology: Star, 6 devices",
        f"- Network seed (controlled comparison): {snapshot['network_seed']}",
        f"- Anchor locations: {json.dumps(snapshot['anchor'], ensure_ascii=False)}",
        f"- Probe dataset: {n_probe_seeds} episodes (rotating 4 fault types, controlled anchor re-inject)",
        "",
        "## 1. Controlled fault comparison (same seed, different fault type)",
        "",
        "| Fault | Location | Discovery Δ cells | Post-scan disc nonzero |",
        "|---|---|---:|---:|",
    ]

    for s in snapshot["samples"]:
        d = s["obs_diff"]
        post = s["obs_post_scan_summary"]
        lines.append(
            f"| {s['fault_type']} | {s['fault_location']} | "
            f"{d['discovery_changed_cells']} | {post['discovery_nonzero']} |"
        )

    lines.extend(["", "### Post-scan L2 distance between fault types (same seed=42)", ""])
    for k, v in snapshot["post_scan_pairwise_l2"].items():
        lines.append(f"- {k}: {v:.4f}")

    lines.extend(
        [
            "",
            "## 2. Scan reveals fault location?",
            "",
            "After `scan_network`, discovery matrix encodes **UP connections only** "
            "(see `env._execute_topology_action` — failed links are omitted, not marked DISCONNECTED).",
            "",
            "Implication: **link failure location** may appear as a **missing edge** among otherwise "
            "connected star edges. Device/performance/misconfig signals in scan-only obs are weaker or absent.",
            "",
            "## 3. Scan-only supervised probe (fault type classification)",
            "",
            f"- Random chance (4-class): {probe.get('random_chance', 0.25):.2%}",
            "",
            "| Model | CV accuracy |",
            "|---|---:|",
        ]
    )

    for name, stats in probe.get("models", {}).items():
        lines.append(f"| {name} | {stats['cv_accuracy_mean']:.1%} ± {stats['cv_accuracy_std']:.1%} |")

    lines.extend(["", "### Per-field logistic regression (post-scan)", "", "| Field | CV accuracy |", "|---|---:|"])
    for field, stats in probe.get("per_field_logistic", {}).items():
        acc = stats.get("cv_accuracy_mean")
        if acc is None:
            lines.append(f"| {field} | {stats.get('note', 'n/a')} |")
        else:
            lines.append(f"| {field} | {acc:.1%} ± {stats.get('cv_accuracy_std', 0):.1%} |")

    lines.extend(
        [
            "",
            "## 4. Interpretation",
            "",
            f"- Best probe: **{interpretation['best_model']}** @ {interpretation['best_cv_accuracy']:.1%}",
            f"- Branch: **{interpretation['branch']}**",
            f"- {interpretation['recommendation']}",
            "",
            "## 5. Next step",
            "",
            "- Branch **A**: pivot metrics/questions toward diagnosis-mapping generalization (DSR, Macro F1, location accuracy).",
            "- Branch **B**: masked multi-fault 100k sanity; ping/status may become necessary.",
            "- Do **not** run link-only 50k/100k expecting ping to emerge.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-seed", type=int, default=42)
    parser.add_argument("--probe-seeds", type=int, default=200)
    parser.add_argument("--probe-start", type=int, default=1000)
    args = parser.parse_args()

    try:
        import sklearn  # noqa: F401
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "-q"])

    print("Collecting controlled 4-fault snapshot...")
    snapshot = controlled_four_fault_snapshot(network_seed=args.network_seed)

    print(f"Building probe dataset ({args.probe_seeds} seeds)...")
    X, y, _ = build_probe_dataset(n_seeds=args.probe_seeds, seed_start=args.probe_start)

    print("Running supervised probe...")
    probe = run_supervised_probe(X, y)
    interpretation = interpret_probe(probe, snapshot["post_scan_pairwise_l2"])

    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    out_json = {
        "snapshot": snapshot,
        "probe": probe,
        "interpretation": interpretation,
        "probe_seeds": args.probe_seeds,
    }
    (results / "post_scan_audit.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    md = build_markdown(snapshot, probe, interpretation, args.probe_seeds)
    report_path = reports / "post_scan_information_audit.md"
    report_path.write_text(md, encoding="utf-8")

    print(f"Branch: {interpretation['branch']} (best CV={interpretation['best_cv_accuracy']:.1%})")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
