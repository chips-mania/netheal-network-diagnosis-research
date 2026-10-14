"""Repeat-action penalty on top of the evidence diagnose-gate.

Train: extra -2 the first time an action_id is reused in the episode.
Eval DSR uses original reward (no extra penalty).
Control = evidence-gate zip already trained (same mask, no repeat penalty).
"""
from __future__ import annotations

import json
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from .env_utils import make_masked_base_env
from .protocol import PROJECT_ROOT, load_protocol
from .repeat_penalty import REPEAT_PENALTY
from .run_evidence_diagnose_gate import _evaluate, _go
from .run_symmetric_reward_poc import set_seeds
from .train_episode_log import find_train_log, summarize_rows


def main() -> None:
    protocol = load_protocol()
    cfg = protocol["repeat_penalty_poc"]
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_repeat_penalty_star.zip"
    gate_zip = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_evidence_gate_star.zip"
    log_path = results / "train_episode_log" / "repeat_penalty_star.jsonl"

    set_seeds(cfg["policy_seed"])
    env = make_masked_base_env(
        cfg["topology_types"],
        protocol=protocol,
        evidence_obs=True,
        evidence_diagnose_gate=True,
        repeat_penalty=True,
        episode_log_path=log_path,
    )
    check_env(env, warn=True)
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, seed=cfg["policy_seed"])
    print(f"RepeatPenalty={REPEAT_PENALTY} + evidence gate + obs wrapper, original terminal reward")
    model.learn(total_timesteps=cfg["timesteps"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt.with_suffix("")))
    logger = find_train_log(env)
    train_sum = summarize_rows(logger.rows) if logger else {}
    env.close()

    n = cfg["eval_episodes"]
    seed_base = cfg["eval_seed_base"]
    print("Eval variant det (gate on, repeat shaping off)...")
    variant = _evaluate(model, protocol, gate=True, n=n, seed_base=seed_base, det=True)
    control = None
    if gate_zip.exists():
        print("Eval evidence-gate control det...")
        ctrl = MaskablePPO.load(str(gate_zip))
        control = _evaluate(ctrl, protocol, gate=True, n=n, seed_base=seed_base, det=True)
    gate = _go(control or {}, variant)

    def _pf(block, ft):
        b = block.get("per_fault", {}).get(ft, {})
        return f"{b.get('dsr', 0):.0%} ({b.get('exact_n', 0)}/{b.get('n', 0)}; cons {b.get('exact_consistent_n', 0)})"

    lines = [
        "# Repeat-action penalty PoC",
        "",
        f"> Train extra {REPEAT_PENALTY} when an action_id is reused in the episode.",
        "> Stacked on evidence diagnose-gate + obs wrapper. Eval reward is original.",
        "> Control = evidence-gate zip (no repeat penalty).",
        "",
        "| Metric | Evidence-gate | +Repeat −2 |",
        "|---|---:|---:|",
    ]
    if control:
        lines += [
            f"| Exact DSR | {control['decomposition']['exact_diagnosis_accuracy']:.0%} | {variant['decomposition']['exact_diagnosis_accuracy']:.0%} |",
            f"| Shortcut | {control['shortcut']['shortcut_rate']:.0%} | {variant['shortcut']['shortcut_rate']:.0%} |",
            f"| AvgSteps | {control['stats']['avg_steps_all']:.1f} | {variant['stats']['avg_steps_all']:.1f} |",
            f"| Timeout | {control['timeout_rate']:.0%} | {variant['timeout_rate']:.0%} |",
            f"| Submit | {control['submit_rate']:.0%} | {variant['submit_rate']:.0%} |",
            f"| Device | {_pf(control, 'device_failure')} | {_pf(variant, 'device_failure')} |",
            f"| Perf | {_pf(control, 'performance_degradation')} | {_pf(variant, 'performance_degradation')} |",
            f"| Misconfig | {_pf(control, 'misconfiguration')} | {_pf(variant, 'misconfiguration')} |",
            f"| Link | {_pf(control, 'link_failure')} | {_pf(variant, 'link_failure')} |",
        ]
    lines += [
        "",
        f"- Train-time exact∧cons\\|tool: {train_sum.get('p_exact_cons_given_tool', 0):.1%}",
        f"- Decision: **{gate['decision']}**",
        f"- {gate['rule']}",
        "",
    ]
    (reports / "repeat_penalty_poc.md").write_text("\n".join(lines), encoding="utf-8")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("records",)}
        return o

    payload = {"train": train_sum, "variant": _clean(variant), "gate": gate}
    if control:
        payload["control"] = _clean(control)
    (results / "repeat_penalty_poc.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"decision={gate['decision']} timeout={gate['timeout_rate']:.0%} shortcut={gate['shortcut']:.0%}")
    print("Wrote", reports / "repeat_penalty_poc.md")


if __name__ == "__main__":
    main()
