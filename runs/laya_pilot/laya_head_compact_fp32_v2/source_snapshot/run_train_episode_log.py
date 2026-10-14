"""Retrain Control-Obs 30k while writing per-episode on-policy logs.

Not a new intervention. Reward/mask unchanged. Checkpoint is tagged `_logged`.
"""
from __future__ import annotations

import json
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from .env_utils import make_masked_base_env
from .protocol import PROJECT_ROOT, load_protocol
from .run_obs_then_e4 import set_seeds
from .train_episode_log import find_train_log, summarize_rows


def _bucket(rows, lo: int, hi: int):
    return [r for r in rows if lo < r["t"] <= hi]


def main() -> None:
    protocol = load_protocol()
    cfg = protocol["obs_then_e4_poc"]
    results = PROJECT_ROOT / "results" / "train_episode_log"
    reports = PROJECT_ROOT / "reports"
    results.mkdir(parents=True, exist_ok=True)
    jsonl = results / "control_obs_star.jsonl"
    summary_path = results / "control_obs_star_summary.json"
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_control_obs_logged_star.zip"

    set_seeds(cfg["policy_seed"])
    env = make_masked_base_env(
        cfg["topology_types"],
        protocol=protocol,
        evidence_obs=True,
        episode_log_path=jsonl,
    )
    check_env(env, warn=True)
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, seed=cfg["policy_seed"])
    print(f"Control-Obs+log ent_coef={model.ent_coef} evidence_obs=True log={jsonl}")
    model.learn(total_timesteps=cfg["timesteps"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt.with_suffix("")))

    logger = find_train_log(env)
    rows = logger.rows if logger else []
    env.close()

    overall = summarize_rows(rows)
    windows = {
        "0-10k": summarize_rows(_bucket(rows, 0, 10000)),
        "10-20k": summarize_rows(_bucket(rows, 10000, 20000)),
        "20-30k": summarize_rows(_bucket(rows, 20000, 40000)),
    }
    payload = {
        "n_episodes": len(rows),
        "timesteps": cfg["timesteps"],
        "evidence_obs": True,
        "overall": overall,
        "windows": windows,
        "jsonl": str(jsonl),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _line(name: str, s: dict) -> str:
        return (
            f"| {name} | {s['n']} | {s['p_tool_use']:.1%} | {s['p_exact']:.1%} | "
            f"{s['p_exact_and_tool']:.1%} | {s['p_exact_given_tool']:.1%} | "
            f"{s['p_exact_cons_given_tool']:.1%} | {s['p_shortcut']:.1%} |"
        )

    pf = overall["per_fault"]
    pf_lines = [
        f"| {ft} | {b['n']} | {b['tool']} | {b['exact']} | {b['exact_tool']} | {b['exact_cons']} |"
        for ft, b in pf.items()
    ]
    report = "\n".join(
        [
            "# Train-time episode log — Control-Obs 30k",
            "",
            "> Same protocol as Stage B. Reward/mask unchanged. Measurement only.",
            f"> JSONL: `{jsonl.relative_to(PROJECT_ROOT)}`",
            "",
            "| window | n | tool-use | exact | exact∧tool | exact\\|tool | exact∧cons\\|tool | shortcut |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            _line("all", overall),
            _line("0–10k", windows["0-10k"]),
            _line("10–20k", windows["10-20k"]),
            _line("20–30k", windows["20-30k"]),
            "",
            "| fault | n | tool | exact | exact∧tool | exact∧cons |",
            "|---|---:|---:|---:|---:|---:|",
            *pf_lines,
            "",
        ]
    )
    (reports / "train_episode_log.md").write_text(report, encoding="utf-8")
    print(json.dumps(overall, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {reports / 'train_episode_log.md'}")


if __name__ == "__main__":
    main()
