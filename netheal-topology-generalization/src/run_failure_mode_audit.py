"""Step 9.11 — Fault-specific failure mode audit (M / D / P)."""
from __future__ import annotations

import argparse
import json

from .failure_mode_audit import (
    audit_device_ppo_type_confusion,
    audit_device_type_confusion,
    audit_misconfig_ppo_type_confusion,
    audit_misconfiguration_location_signal,
    audit_performance_location,
)
from .protocol import PROJECT_ROOT, load_protocol


def build_report(m_audit: dict, d_audit: dict, p_audit: dict, m_ppo: dict, d_ppo: dict) -> str:
    m_probe = m_audit["post_scan_location_probe"]
    d_type = d_audit["type_probe_after_diagnostics"]
    p_scan = p_audit["post_scan_location_probe"]
    p_iface = p_audit["post_interfaces_location_probe"]

    lines = [
        "# Step 9.11 — Fault-Specific Failure Mode Audit",
        "",
        "Purpose: explain **how** each non-link fault fails (type vs location), now that 9.10 ruled out missing evidence.",
        "",
        "## Summary",
        "",
        "| Fault | PPO failure mode | Root cause hypothesis |",
        "|---|---|---|",
        "| Link | none (DSR 100%) | scan exposes missing edge |",
        "| Device | type mapping | evidence present after tools; policy maps to performance_degradation |",
        "| Performance | location | type learned; scan-only shortcut to fixed edge_1→center |",
        "| Misconfiguration | type discrimination | scan exposes location (missing link); wrong diagnosis category |",
        "",
        "---",
        "",
        "## 9.11-M — Misconfiguration: why location = 100%?",
        "",
        f"- Post-scan location probe CV accuracy: **{m_probe.get('cv_accuracy', 'n/a')}**",
        "",
        "### Scan discovery pattern (vs healthy baseline)",
        "",
        "| Fault location | Healthy nonzero cells | Fault nonzero cells |",
        "|---|---:|---:|",
    ]

    for p in m_audit["scan_patterns"]:
        lines.append(
            f"| `{p['fault_location']}` | {p['healthy_discovery_nonzero']} | {p['fault_discovery_nonzero']} |"
        )

    lines.extend(
        [
            "",
            m_audit["interpretation"],
            "",
            "### PPO wrong-type pattern (from eval trace)",
            "",
            f"- Wrong-type predictions by diagnosis category: {m_ppo['wrong_type_predictions']}",
            f"- Dominant wrong patterns (actual loc, pred type, pred target):",
        ]
    )
    for key, cnt in m_ppo["dominant_wrong_patterns"].items():
        lines.append(f"  - `{key}`: {cnt}")
    lines.append(f"- {m_ppo['note']}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 9.11-D — Device: why type → performance_degradation?",
            "",
            f"- Post-scan obs identical for device vs perf at anchor: **{d_audit['post_scan_obs_identical']}**",
            f"- Obs distance after diagnostics (device vs perf): **{d_audit['obs_distance_device_vs_perf']:.3f}**",
            f"- Type probe after diagnostics (2-class): **{d_type.get('cv_accuracy', 'n/a')}**",
            "",
            "### Key tool results (device_failure @ edge_0)",
            "",
            f"- ping center→edge_0: {d_audit['device_key_tool_results'].get('ping_center_to_anchor')}",
            f"- check_status edge_0: {d_audit['device_key_tool_results'].get('status_anchor')}",
            f"- check_interfaces edge_0: {d_audit['device_key_tool_results'].get('interfaces_anchor')}",
            "",
            f"- PPO predicted types: {d_ppo['predicted_type_distribution']}",
            f"- {d_ppo['note']}",
            "",
            d_audit["interpretation"],
            "",
            "---",
            "",
            "## 9.11-P — Performance: why location guess?",
            "",
            f"- Post-scan location probe: **{p_scan.get('cv_accuracy', 'n/a')}** (expected low)",
            f"- Post check_interfaces location probe: **{p_iface.get('cv_accuracy', 'n/a')}**",
            f"- PPO fixed guess: **{p_audit['ppo_fixed_guess']}**",
            "",
            "### Interface latency signatures (center, by faulted link)",
            "",
        ]
    )
    for sig in p_audit["interface_latency_signatures"]:
        step = sig.get("center_interfaces") or {}
        data = step.get("tool_data", {})
        lines.append(f"- fault `{sig['fault_location']}`: interfaces={data.get('interfaces', [])}")

    lines.extend(["", p_audit["interpretation"], ""])

    lines.extend(
        [
            "---",
            "",
            "## Implications before 100k / curriculum",
            "",
            "1. **Misconfiguration**: teach/limit diagnosis mapping — location signal is scan-like (missing edge).",
            "2. **Device**: reward or action-cost may favor performance_degradation diagnosis despite device-down tools.",
            "3. **Performance**: must incentivize interface/traceroute use; scan→diagnose cannot recover location.",
            "",
            "100k is justified only as **targeted** experiments per failure mode, not blind scaling.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-seed", type=int, default=42)
    args = parser.parse_args()

    protocol = load_protocol()
    results_dir = PROJECT_ROOT / "results"
    reports_dir = PROJECT_ROOT / "reports"

    print("9.11-M misconfiguration location signal...")
    m_audit = audit_misconfiguration_location_signal(protocol, network_seed=args.network_seed)

    print("9.11-D device type confusion...")
    d_audit = audit_device_type_confusion(protocol, network_seed=args.network_seed)

    print("9.11-P performance location...")
    p_audit = audit_performance_location(protocol, network_seed=args.network_seed)

    m_ppo = audit_misconfig_ppo_type_confusion(results_dir / "misconfiguration_only_action_trace.csv")
    d_ppo = audit_device_ppo_type_confusion(results_dir / "device_failure_only_action_trace.csv")

    report = build_report(m_audit, d_audit, p_audit, m_ppo, d_ppo)
    report_path = reports_dir / "failure_mode_audit.md"
    report_path.write_text(report, encoding="utf-8")

    payload = {
        "misconfiguration": m_audit,
        "device": d_audit,
        "performance": p_audit,
        "misconfiguration_ppo": m_ppo,
        "device_ppo": d_ppo,
    }
    (results_dir / "failure_mode_audit.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
