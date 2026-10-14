"""Step 5: fault injection sanity."""
from __future__ import annotations

import argparse
from pathlib import Path

from netheal.faults.injector import FaultType

from .env_utils import make_base_env
from .protocol import PROJECT_ROOT, load_protocol


def fault_sanity_report() -> str:
    protocol = load_protocol()
    lines = ["# Fault Injection Sanity", ""]
    for topo in ["star", "linear", "mesh"]:
        lines.append(f"## {topo}")
        for ft_name in protocol["fault_types"]:
            fault = FaultType(ft_name)
            env = make_base_env([topo], fault_types=[fault], protocol=protocol)
            _, info = env.reset(seed=42)
            gt = info.get("ground_truth_fault")
            lines.append(f"- {ft_name}: {gt}")
            env.close()
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    out = PROJECT_ROOT / "reports" / "fault_sanity.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fault_sanity_report(), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
