"""Step 9.9 — Non-link single-fault learnability audit (D / P / M)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from netheal.faults.injector import FaultType

from .action_trace import write_trace_csv
from .diagnosis_decomposition import format_matching_table
from .protocol import PROJECT_ROOT, load_protocol
from .single_fault_sanity import (
    FAULT_ALIASES,
    eval_single_fault_model,
    model_path_for_fault,
    train_single_fault,
)


def overall_verdict(results: list) -> dict:
    verdicts = [r["learnability"]["verdict"] for r in results]
    valid = sum(1 for v in verdicts if v == "valid_troubleshooting_policy")
    partial = sum(1 for v in verdicts if v in ("partial_or_lucky", "high_dsr_shortcut"))
    no_policy = sum(1 for v in verdicts if v == "no_valid_policy_30k")

    if valid == 3:
        case = "valid_policy_all_three"
        next_step = "Proceed to masked multi-fault 100k sanity."
    elif valid >= 1:
        case = "mixed_policy_signal"
        next_step = "Run Step 9.10 solvability audit on failing faults before multi-fault 100k."
    else:
        case = "no_valid_policy_under_current_settings"
        next_step = (
            "Do not run multi-fault 100k yet. Run Step 9.10 solvability audit to separate "
            "environment evidence limits from RL learning limits."
        )

    return {
        "case": case,
        "valid_policy_count": valid,
        "partial_count": partial,
        "no_valid_policy_count": no_policy,
        "next_step": next_step,
        "wording": (
            "Under current settings (MaskablePPO, 30k, seed 1), no valid troubleshooting policy "
            "was learned for the three non-link faults. This does NOT yet prove the faults are "
            "unsolvable — only that PPO did not learn effective policies in this run."
        ),
    }


def build_report(results: list, link_reference: dict, overall: dict, n_episodes: int) -> str:
    lines = [
        "# Step 9.9 — Non-link Single-Fault Learnability Audit",
        "",
        "Config: Star, 6 devices, Hint OFF, MaskablePPO, max_steps=15, 30k steps, seed 1",
        "",
        f"Eval episodes per model: {n_episodes}",
        "",
        "> **Interpretation note:** Results show whether PPO learned a valid troubleshooting policy "
        "under this specific setup. They do **not** by themselves prove that a fault is unsolvable in NetHeal.",
        "",
        "## Reference (Step 9.7 Link-only)",
        "",
        f"- Exact diagnosis (DSR): {link_reference['dsr']:.0%}, AvgSteps: {link_reference['avg_steps']:.1f}, "
        f"behavior: scan → diagnose (scan-only observability)",
        "",
        "## Summary table",
        "",
        "| Model | Fault | DSR | Type acc | Loc acc | AvgSteps | Tools before dx | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for r in results:
        s = r["ppo_stats"]
        d = r["diagnosis_decomposition"]
        l = r["learnability"]
        lines.append(
            f"| {r['alias']} | {r['fault_type']} | {s['dsr']:.0%} | "
            f"{d['fault_type_accuracy']:.0%} | {d['location_accuracy']:.0%} | "
            f"{s['avg_steps_all']:.1f} | {l['tool_before_diagnosis']:.1f} | **{l['verdict']}** |"
        )

    for r in results:
        b = r["behavior"]
        l = r["learnability"]
        d = r["diagnosis_decomposition"]
        lines.extend(
            [
                "",
                f"## Model {r['alias']} — {r['fault_type']}",
                "",
                "### Diagnosis decomposition",
                "",
                f"- Fault-type accuracy: {d['fault_type_accuracy']:.0%}",
                f"- Location accuracy: {d['location_accuracy']:.0%}",
                f"- Exact diagnosis (DSR): {d['exact_diagnosis_accuracy']:.0%}",
                f"- DSR margin vs Random: {l['dsr_margin_vs_random']:.1%}",
                "",
                f"- First-step diagnosis: {b['first_action_diagnosis_rate']:.0%}",
                f"- scan: {b['scan_usage_rate']:.0%}, ping: {b['ping_usage_rate']:.0%}, "
                f"traceroute: {b['traceroute_usage_rate']:.0%}, "
                f"status: {b['check_status_usage_rate']:.0%}, "
                f"interfaces: {b['check_interfaces_usage_rate']:.0%}",
                f"- Action types (non-diagnosis): {b.get('action_type_frequency', {})}",
                "",
                "### Episode-level matching (first 20)",
                "",
            ]
        )
        lines.extend(format_matching_table(d["rows"], limit=20))

        if r["fault_type"] == "misconfiguration":
            loc_only_hits = sum(
                1 for row in d["rows"] if row["location_correct"] and not row["exact_correct"]
            )
            lines.extend(
                [
                    "",
                    "### Misconfiguration note",
                    "",
                    f"- Episodes with location correct but wrong type: {loc_only_hits}",
                    "- Compare aggregate location counts vs episode table before inferring observability.",
                ]
            )

        if r["success_traces"]:
            lines.extend(["", "### Representative successful traces", ""])
            for trace in r["success_traces"]:
                lines.append("```text")
                lines.append(trace)
                lines.append("```")

    lines.extend(
        [
            "",
            "## Overall verdict",
            "",
            f"- **{overall['case']}**",
            f"- {overall['wording']}",
            f"- Next: {overall['next_step']}",
            "",
            "### Decision guide",
            "",
            "- **valid_policy_all_three**: valid troubleshooting per fault → masked multi-fault 100k justified.",
            "- **mixed_policy_signal**: solvability audit on failing faults before joint training.",
            "- **no_valid_policy_under_current_settings**: run Step 9.10 before changing reward/curriculum/100k.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--seed-base", type=int, default=80000)
    args = parser.parse_args()

    protocol = load_protocol()
    cfg = protocol["nonlink_single_fault_sanity"]
    out_dir = PROJECT_ROOT / "models" / "checkpoints"
    reports = PROJECT_ROOT / "reports"
    results_dir = PROJECT_ROOT / "results"

    fault_names = cfg["fault_types"]
    all_results = []

    for fault_name in fault_names:
        fault = FaultType(fault_name)
        path = model_path_for_fault(fault_name, out_dir)
        print(f"\n=== {FAULT_ALIASES.get(fault_name, '?')} — {fault_name} ===")
        if args.skip_train and path.with_suffix(".zip").exists():
            model = MaskablePPO.load(str(path.with_suffix("")))
        elif args.skip_train:
            print(f"Missing model {path}, skipping train flag inconsistent")
            continue
        else:
            print(f"Training {fault_name} only (30k)...")
            model = train_single_fault(protocol, fault, out_dir)

        result = eval_single_fault_model(
            model,
            protocol,
            fault,
            n_episodes=args.eval_episodes,
            seed_base=args.seed_base,
        )
        trace_path = results_dir / f"{fault_name}_only_action_trace.csv"
        write_trace_csv(trace_path, result.pop("records"))
        all_results.append(result)
        d = result["diagnosis_decomposition"]
        print(
            f"  DSR={result['ppo_stats']['dsr']:.0%} "
            f"Type={d['fault_type_accuracy']:.0%} Loc={d['location_accuracy']:.0%} "
            f"Verdict={result['learnability']['verdict']}"
        )

    link_ref = {"dsr": 1.0, "avg_steps": 2.0}
    overall = overall_verdict(all_results)

    report = build_report(all_results, link_ref, overall, args.eval_episodes)
    report_path = reports / "nonlink_single_fault_learnability.md"
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "results": [
            {
                k: v
                for k, v in r.items()
                if k not in ("success_traces", "diagnosis_decomposition")
            }
            for r in all_results
        ],
        "diagnosis_decomposition": {
            r["fault_type"]: {
                "fault_type_accuracy": r["diagnosis_decomposition"]["fault_type_accuracy"],
                "location_accuracy": r["diagnosis_decomposition"]["location_accuracy"],
                "exact_diagnosis_accuracy": r["diagnosis_decomposition"]["exact_diagnosis_accuracy"],
            }
            for r in all_results
        },
        "overall": overall,
        "link_reference": link_ref,
    }
    (results_dir / "nonlink_single_fault_learnability.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nOverall: {overall['case']}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
