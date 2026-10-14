"""Step 9.12 — Diagnosis action / reward factorization audit runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from sb3_contrib import MaskablePPO

from netheal.faults.injector import FaultType

from .diagnosis_factorization_audit import audit_ppo_diagnosis_marginals, run_full_audit
from .protocol import PROJECT_ROOT, load_protocol
from .single_fault_sanity import model_path_for_fault


def _reward_table(grid: dict) -> List[str]:
    lines = [
        f"### {grid['ground_truth_type']} @ `{grid['ground_truth_location']}`",
        "",
        "| Case | Type OK | Loc OK | Exact | Total reward | diagnosis | partial_device |",
        "|---|:---:|:---:|:---:|---:|---:|---:|",
    ]
    for row in grid["cases"]:
        b = row["breakdown"]
        lines.append(
            f"| {row['case']} | {'Y' if row['type_correct'] else 'N'} | "
            f"{'Y' if row['location_correct'] else 'N'} | "
            f"{'Y' if row['exact_correct'] else 'N'} | "
            f"{row['total_reward']:.2f} | {b.get('diagnosis_reward', 0):.2f} | "
            f"{b.get('partial_device_reward', 0):.2f} |"
        )
    lines.append("")
    return lines


def build_report(audit: dict, ppo_marginals: dict) -> str:
    asp = audit["action_space"]
    ps = audit["post_scan_valid_actions"]
    lines = [
        "# Step 9.12 — Diagnosis Action / Reward Factorization Audit",
        "",
        "Purpose: test whether failures are driven by **type×location joint diagnosis** and",
        "**asymmetric sparse reward**, before any fault-specific reward shaping.",
        "",
        "## 9.12-A — Diagnosis action structure",
        "",
        f"- Devices: {asp['n_devices']} → `{asp['devices']}`",
        f"- Total action space: **{asp['total_action_space']}**",
        f"- Diagnosis actions: **{asp['diagnosis_action_count']}** "
        f"(expected fault×location product: {asp['expected_diagnosis_count']})",
        f"- Is pure type×location Cartesian product: **{asp['is_fault_type_x_location_product']}**",
        "",
        "### Diagnosis count by fault type",
        "",
    ]
    for ft, cnt in asp["diagnosis_by_fault_type"].items():
        lines.append(f"- {ft}: {cnt}")
    lines.extend(
        [
            "",
            f"- Device-level locations: {asp['unique_device_locations']}",
            f"- Link-level locations: {asp['unique_link_locations']}",
            "",
            "Full mapping: `results/diagnosis_action_mapping.json`",
            "",
            "## 9.12-B — Post-scan valid actions (masking, leakage-free)",
            "",
            f"| Phase | Total valid | topology | diagnostic | diagnosis |",
            "|---|---:|---:|---:|---:|",
            f"| pre-scan | {ps['pre_scan_valid_total']} | "
            f"{ps['pre_scan_by_category'].get('topology_discovery', 0)} | "
            f"{ps['pre_scan_by_category'].get('diagnostic', 0)} | "
            f"{ps['pre_scan_by_category'].get('diagnosis', 0)} |",
            f"| post-scan | {ps['post_scan_valid_total']} | "
            f"{ps['post_scan_by_category'].get('topology_discovery', 0)} | "
            f"{ps['post_scan_by_category'].get('diagnostic', 0)} | "
            f"**{ps['post_scan_by_category'].get('diagnosis', 0)}** |",
            "",
            f"- Post-scan **{ps['post_scan_diagnosis_count']}** terminal diagnosis candidates immediately available.",
            f"- By type: {ps['post_scan_diagnosis_by_type']}",
            "",
            "> Scan opens the full diagnosis lottery while tools remain optional — PPO can skip evidence gathering.",
            "",
            "## 9.12-C — Reward factorization (4-way grid @ anchor)",
            "",
            f"Scaling multiplier (network_size={audit['reward_grids'][list(audit['reward_grids'].keys())[0]]['network_size']}): "
            f"**{audit['reward_grids'][list(audit['reward_grids'].keys())[0]]['scaling_multiplier']:.1f}**",
            "",
        ]
    )

    for ft, grid in audit["reward_grids"].items():
        lines.extend(_reward_table(grid))

    lines.extend(
        [
            "### Reward asymmetry (key finding)",
            "",
            "- **Exact match** (type AND location): large positive reward (~+15.9 at 6 devices).",
            "- **Wrong type + correct location**: often **partial_device_reward** (location overlap), not full penalty.",
            "- **Correct type + wrong location**: partial_device_reward if device overlap exists.",
            "- **No type-only partial reward** — type correctness alone is never rewarded without location overlap.",
            "- This asymmetry can produce **complementary failures**: learn location OR type shortcut separately.",
            "",
            "## 9.12-D — PPO diagnosis probability marginals (post-scan)",
            "",
        ]
    )

    for ft, data in ppo_marginals.items():
        if "error" in data:
            lines.append(f"- **{ft}**: {data['error']}")
            continue
        lines.extend(
            [
                f"### {ft}",
                "",
                f"- mean P(correct type): **{data['mean_p_correct_type']:.3f}**",
                f"- mean P(correct location): **{data['mean_p_correct_location']:.3f}**",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## Interpretation / decision tree",
            "",
            "```",
            "Evidence sufficient (9.10)",
            "        ↓",
            "Diagnosis = type × location joint discrete action (~72 on Star)",
            "        ↓",
            "Scan opens ~72 terminal diagnoses immediately",
            "        ↓",
            "Reward: exact match only full credit; partial credit is location-biased",
            "        ↓",
            "PPO learns type-only OR location-only shortcuts (9.9/9.11)",
            "        ↓",
            "Sparse joint classification problem — NOT fault-specific missing info",
            "```",
            "",
            "### Before intervention",
            "",
            "- Do **not** apply fault-specific reward shaping (M type bonus, P interface bonus, etc.).",
            "- Prefer **generic** fixes applied uniformly: partial type reward, factorized diagnosis actions,",
            "  or curriculum — each modifies benchmark semantics; tag as protocol variant.",
            "- Blind 100k alone is unlikely to fix joint type×location binding without structural change.",
            "",
            "### Next decision (after confirming this audit)",
            "",
            "1. Generic partial type+location reward (uniform all faults)",
            "2. Factorized diagnosis action (type → location)",
            "3. Restrict post-scan diagnosis until diagnostics used",
            "4. Pivot research framing toward RL failure-mode analysis",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-seed", type=int, default=42)
    parser.add_argument("--anchor", type=str, default="edge_0")
    parser.add_argument("--ppo-episodes", type=int, default=20)
    args = parser.parse_args()

    protocol = load_protocol()
    out_dir = PROJECT_ROOT / "models" / "checkpoints"
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"

    print("Running 9.12 action/reward factorization audit...")
    audit = run_full_audit(protocol, network_seed=args.network_seed, anchor=args.anchor)

    (results / "diagnosis_action_mapping.json").write_text(
        json.dumps(audit["action_space"]["mapping_table"], indent=2),
        encoding="utf-8",
    )

    ppo_marginals = {}
    for ft_name in ("misconfiguration", "performance_degradation", "device_failure"):
        path = model_path_for_fault(ft_name, out_dir)
        if not path.with_suffix(".zip").exists():
            ppo_marginals[ft_name] = {"error": f"missing checkpoint {path}"}
            continue
        print(f"PPO marginals for {ft_name}...")
        model = MaskablePPO.load(str(path.with_suffix("")))
        ppo_marginals[ft_name] = audit_ppo_diagnosis_marginals(
            model,
            protocol,
            FaultType(ft_name),
            n_episodes=args.ppo_episodes,
        )

    report = build_report(audit, ppo_marginals)
    report_path = reports / "diagnosis_factorization_audit.md"
    report_path.write_text(report, encoding="utf-8")

    payload = {
        "audit": {
            k: v
            for k, v in audit.items()
            if k != "action_space"
        },
        "action_space_summary": {
            k: v for k, v in audit["action_space"].items() if k != "mapping_table"
        },
        "ppo_marginals_summary": {
            k: {
                key: v[key]
                for key in ("mean_p_correct_type", "mean_p_correct_location", "n_episodes", "error")
                if key in v
            }
            for k, v in ppo_marginals.items()
        },
    }
    (results / "diagnosis_factorization_audit.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
