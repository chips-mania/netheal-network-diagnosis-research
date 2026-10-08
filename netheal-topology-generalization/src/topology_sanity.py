"""Step 3-4: observation and topology sanity checks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from netheal import NetworkTroubleshootingEnv

from .protocol import PROJECT_ROOT, load_protocol


def check_observation_shapes() -> str:
    protocol = load_protocol()
    lines = ["## Observation shape check\n"]
    for topo in ["star", "linear", "mesh"]:
        env = NetworkTroubleshootingEnv(
            min_devices=protocol["min_devices"],
            max_devices=protocol["max_devices"],
            max_episode_steps=protocol["max_episode_steps"],
            topology_types=[topo],
            enable_user_hints=protocol["enable_user_hints"],
        )
        obs, _ = env.reset(seed=42)
        shapes = {k: v.shape for k, v in obs.items()}
        lines.append(f"- **{topo}**: {shapes}")
        env.close()
    return "\n".join(lines) + "\n"


def mesh_stats(seed: int, samples: int = 10) -> dict:
    protocol = load_protocol()
    env = NetworkTroubleshootingEnv(
        min_devices=protocol["min_devices"],
        max_devices=protocol["max_devices"],
        max_episode_steps=protocol["max_episode_steps"],
        topology_types=["mesh"],
        enable_user_hints=protocol["enable_user_hints"],
    )
    device_counts = []
    edge_counts = []
    for i in range(samples):
        _, info = env.reset(seed=seed + i)
        net = env.network
        device_counts.append(info["network_size"])
        edge_counts.append(len(net.get_all_connections()))
    env.close()
    return {
        "seed": seed,
        "devices_mean": float(np.mean(device_counts)),
        "devices_std": float(np.std(device_counts)),
        "edges_mean": float(np.mean(edge_counts)),
        "edges_std": float(np.std(edge_counts)),
        "devices_samples": device_counts,
        "edges_samples": edge_counts,
    }


def topology_sanity_report() -> str:
    protocol = load_protocol()
    lines = [
        "# Topology Sanity Report",
        "",
        check_observation_shapes(),
        "## Fixed 6-device check (10 samples each)",
        "",
    ]
    for topo in ["star", "linear", "mesh"]:
        env = NetworkTroubleshootingEnv(
            min_devices=6,
            max_devices=6,
            topology_types=[topo],
            enable_user_hints=False,
        )
        sizes = []
        edges = []
        for s in range(10):
            _, info = env.reset(seed=1000 + s)
            sizes.append(info["network_size"])
            edges.append(len(env.network.get_all_connections()))
        env.close()
        lines.append(
            f"- **{topo}**: devices={sizes} (unique={set(sizes)}), "
            f"edges mean={np.mean(edges):.1f}"
        )

    lines.extend(["", "## Mesh instances M1/M2/M3", ""])
    for label, seed in protocol["mesh_instances"].items():
        stats = mesh_stats(seed, samples=5)
        lines.append(
            f"- **{label}** (seed={seed}): devices={stats['devices_samples']}, "
            f"edges={stats['edges_samples']}"
        )

    lines.extend(
        [
            "",
            "## Hierarchical (record only — excluded from Primary)",
            "",
            "Default hierarchical generator ignores min/max_devices; not used in Primary.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "reports" / "topology_sanity.md",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(topology_sanity_report(), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
