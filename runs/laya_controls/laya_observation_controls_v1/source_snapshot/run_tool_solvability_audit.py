"""Step 9.10 — Solvability / Tool Information Audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from netheal.faults.injector import FaultType

from .protocol import PROJECT_ROOT, load_protocol
from .tool_solvability_audit import run_fault_type_audit, run_location_audit


def _solvability_table(audit_a: dict) -> str:
    lines = [
        "## 9.10-A — Tool information vs healthy baseline (anchor edge_0)",
        "",
        "| Fault | Scan | Ping | Status | Interface | Traceroute | Evidence exists? |",
        "|---|:---:|:---:|:---:|:---:|:---:|---|",
    ]
    for fault, data in audit_a["per_fault"].items():
        disc = data["discriminability"]
        per = disc["per_tool"]
        cells = []
        for tool in ("scan", "ping", "status", "interfaces", "traceroute"):
            cells.append("Y" if per[tool]["differs_from_healthy"] else "N")
        verdict = disc["verdict"]
        useful = ", ".join(disc["useful_tools"]) if disc["useful_tools"] else "none"
        lines.append(f"| {fault} | {' | '.join(cells)} | **{verdict}** ({useful}) |")
    lines.extend(
        [
            "",
            "### Fault-type separability at same anchor",
            "",
            f"- Verdict: **{audit_a['fault_type_discriminability']['verdict']}**",
            f"- Separable pairs: {audit_a['fault_type_discriminability'].get('separable_pairs', [])}",
        ]
    )
    return "\n".join(lines)


def _location_table(audit_b: dict) -> str:
    ft = audit_b["fault_type"]
    loc_disc = audit_b["location_discriminability"]
    lines = [
        f"## 9.10-B — Location discriminability ({ft})",
        "",
        f"- Locations tested: {audit_b['locations']}",
        f"- Verdict: **{loc_disc['verdict']}**",
        f"- Differing pairs: {loc_disc.get('differing_pairs', [])}",
        "",
    ]
    if loc_disc.get("pairwise_tool_diffs"):
        lines.append("### Pairwise tool diffs")
        lines.append("")
        for pair, diffs in loc_disc["pairwise_tool_diffs"].items():
            changed = [t for t, v in diffs.items() if v]
            lines.append(f"- `{pair}`: {changed or 'no tool diff'}")
    return "\n".join(lines)


def _decision_tree(audit_a: dict, location_audits: list) -> str:
    sufficient_faults = [
        ft for ft, d in audit_a["per_fault"].items() if d["discriminability"]["verdict"] == "sufficient"
    ]
    partial_faults = [
        ft for ft, d in audit_a["per_fault"].items() if d["discriminability"]["verdict"] == "partial"
    ]
    none_faults = [
        ft for ft, d in audit_a["per_fault"].items() if d["discriminability"]["verdict"] == "none"
    ]

    loc_sep = {a["fault_type"]: a["location_discriminability"]["verdict"] for a in location_audits}

    lines = [
        "## Decision guide",
        "",
        "### A. Sufficient tool evidence (fault vs healthy)",
        "",
        f"- Sufficient: {sufficient_faults or 'none'}",
        f"- Partial: {partial_faults or 'none'}",
        f"- None: {none_faults or 'none'}",
        "",
        "### B. Location separability",
        "",
    ]
    for ft, verdict in loc_sep.items():
        lines.append(f"- {ft}: **{verdict}**")

    lines.extend(
        [
            "",
            "### Interpretation",
            "",
            "- **A + location separable + PPO failed (9.9)** → RL learning problem (100k / curriculum / reward).",
            "- **Partial evidence** → task definition / observation design issue.",
            "- **No evidence** → benchmark limitation; exclude fault from primary RL experiment.",
        ]
    )
    return "\n".join(lines)


def build_report(audit_a: dict, location_audits: list) -> str:
    parts = [
        "# Step 9.10 — Solvability / Tool Information Audit",
        "",
        "Purpose: determine whether NetHeal tools expose enough evidence to identify fault type and location,",
        "independent of PPO training.",
        "",
        f"Network seed: {audit_a['network_seed']}, anchor: {audit_a['anchor_location']}",
        "",
        _solvability_table(audit_a),
        "",
    ]
    for audit_b in location_audits:
        parts.append(_location_table(audit_b))
        parts.append("")
    parts.append(_decision_tree(audit_a, location_audits))
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-seed", type=int, default=42)
    parser.add_argument("--anchor", type=str, default="edge_0")
    args = parser.parse_args()

    protocol = load_protocol()
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"

    print("Running 9.10-A fault-type tool audit...")
    audit_a = run_fault_type_audit(protocol, network_seed=args.network_seed, anchor_location=args.anchor)

    location_audits = []
    for ft_name in ("device_failure", "performance_degradation", "misconfiguration"):
        print(f"Running 9.10-B location audit for {ft_name}...")
        location_audits.append(
            run_location_audit(protocol, FaultType(ft_name), network_seed=args.network_seed)
        )

    report = build_report(audit_a, location_audits)
    report_path = reports / "tool_solvability_audit.md"
    report_path.write_text(report, encoding="utf-8")

    payload = {
        "audit_a": audit_a,
        "audit_b": location_audits,
    }
    (results / "tool_solvability_audit.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
