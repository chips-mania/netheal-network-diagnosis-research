"""Step 6: runtime benchmark to validate protocol episode counts."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .protocol import PROJECT_ROOT, load_protocol
from .run_baseline import iter_eval_jobs, random_valid_fn
from .env_utils import make_eval_env


def main() -> None:
    protocol = load_protocol()
    jobs = list(iter_eval_jobs(protocol))
    n = min(20, len(jobs))
    t0 = time.perf_counter()
    for group, topo, mesh_label, seed, fault in jobs[:n]:
        env = make_eval_env([topo], fault_types=[fault], protocol=protocol)
        random_valid_fn(env, seed)
        env.close()
    elapsed = time.perf_counter() - t0
    per_ep = elapsed / n
    total = len(jobs)

    report = (
        f"# Runtime Benchmark\n\n"
        f"- Sample episodes: {n}\n"
        f"- Elapsed: {elapsed:.2f}s\n"
        f"- Per episode: {per_ep:.3f}s\n"
        f"- Full suite size (current protocol): {total} episodes\n"
        f"- Estimated full baseline runtime: {per_ep * total / 60:.1f} min\n"
        f"- episodes_per_fault: {protocol['episodes_per_fault']}\n"
    )
    out = PROJECT_ROOT / "reports" / "runtime_benchmark.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
