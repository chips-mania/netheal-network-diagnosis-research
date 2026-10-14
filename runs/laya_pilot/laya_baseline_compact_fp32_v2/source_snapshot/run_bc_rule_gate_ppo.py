"""Full Rule-BC (in-process) + evidence diagnose-gate + MaskablePPO.

Named variant. Trains full Rule-BC then gated PPO in one process
(canonical star pad=6) so action IDs stay aligned with demos.
"""
from __future__ import annotations

import json

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from .env_utils import make_masked_base_env
from .evidence_diagnose_gate import smoke_test
from .protocol import PROJECT_ROOT, load_protocol
from .run_bc_seed_gate_ppo import _evaluate_pad_gate
from .run_evidence_diagnose_gate import _go
from .run_star_scale_poc import _exact, _pf, bc_train, collect_rule_demos, evaluate_model
from .run_symmetric_reward_poc import set_seeds
from .train_episode_log import find_train_log, summarize_rows


def main() -> None:
    protocol = load_protocol()
    cfg = protocol.setdefault(
        "bc_rule_gate_ppo_poc",
        {
            "topology_types": ["star"],
            "n_demos": 2000,
            "bc_epochs": 40,
            "timesteps": 30000,
            "policy_seed": 1,
            "eval_episodes": 100,
            "eval_seed_base": 112000,
            "pad": 6,
        },
    )
    protocol.setdefault("star_scale_poc", {"topology_types": ["star"]})
    protocol["star_scale_poc"]["topology_types"] = cfg["topology_types"]
    # _evaluate_pad_gate reads bc_seed_gate_ppo_poc topology_types
    protocol.setdefault("bc_seed_gate_ppo_poc", {"topology_types": ["star"]})
    protocol["bc_seed_gate_ppo_poc"]["topology_types"] = cfg["topology_types"]

    smoke = smoke_test()
    print("smoke", smoke)
    if not all(smoke.values()):
        raise SystemExit(f"smoke failed: {smoke}")

    set_seeds(int(cfg["policy_seed"]))
    topologies = cfg["topology_types"]
    pad = int(cfg.get("pad", 6))
    n_dev = 6
    n = int(cfg["eval_episodes"])
    seed_base = int(cfg["eval_seed_base"])

    print(f"=== Full Rule-BC demos={cfg['n_demos']} epochs={cfg['bc_epochs']} pad={pad} ===")
    obs, acts, masks, collect = collect_rule_demos(
        protocol,
        n_devices=n_dev,
        pad=pad,
        n_ok=int(cfg["n_demos"]),
        seed0=20_000,
    )
    print("collect", collect)
    if collect["ok"] < 500:
        raise SystemExit(f"too few demos: {collect}")

    env_bc = make_masked_base_env(
        topologies,
        protocol=protocol,
        evidence_obs=True,
        evidence_diagnose_gate=False,
        min_devices=n_dev,
        max_devices=n_dev,
        star_pad_max=pad,
    )
    check_env(env_bc, warn=True)
    model = MaskablePPO("MultiInputPolicy", env_bc, verbose=0, seed=int(cfg["policy_seed"]))
    losses = bc_train(
        model,
        obs,
        acts,
        masks,
        epochs=int(cfg["bc_epochs"]),
        batch=64,
        lr=3e-4,
    )
    seed_ckpt = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_bc_rule_pad6_pre_gate.zip"
    seed_ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(seed_ckpt.with_suffix("")))
    env_bc.close()

    print("Eval full BC (gate OFF, in-process)...")
    seed_off = evaluate_model(
        model, protocol, n_devices=n_dev, pad=pad, n=n, seed_base=seed_base, det=True
    )
    print(
        f"  exact={_exact(seed_off):.0%} shortcut={seed_off['shortcut']['shortcut_rate']:.0%} "
        f"timeout={seed_off['timeout_rate']:.0%}"
    )

    print("Eval full BC zero-shot under gate ON...")
    seed_on = _evaluate_pad_gate(
        model, protocol, pad=pad, n=n, seed_base=seed_base + 50, det=True, gate=True, n_devices=n_dev
    )
    print(
        f"  exact={_exact(seed_on):.0%} shortcut={seed_on['shortcut']['shortcut_rate']:.0%} "
        f"timeout={seed_on['timeout_rate']:.0%} submit={seed_on['submit_rate']:.0%}"
    )

    log_path = PROJECT_ROOT / "results" / "train_episode_log" / "bc_rule_gate_ppo_star.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = make_masked_base_env(
        topologies,
        protocol=protocol,
        evidence_obs=True,
        evidence_diagnose_gate=True,
        episode_log_path=log_path,
        min_devices=n_dev,
        max_devices=n_dev,
        star_pad_max=pad,
    )
    check_env(env, warn=True)
    # keep same in-process weights; set env for learn()
    model.set_env(env)
    print(f"=== PPO + evidence gate {cfg['timesteps']} steps (from full Rule-BC) ===")
    model.learn(total_timesteps=int(cfg["timesteps"]))
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_bc_rule_gate_ppo_star.zip"
    model.save(str(ckpt.with_suffix("")))
    logger = find_train_log(env)
    train_sum = summarize_rows(logger.rows) if logger else {}
    env.close()

    print("Eval after PPO (gate ON)...")
    variant = _evaluate_pad_gate(
        model, protocol, pad=pad, n=n, seed_base=seed_base + 100, det=True, gate=True, n_devices=n_dev
    )
    print(
        f"  exact={_exact(variant):.0%} shortcut={variant['shortcut']['shortcut_rate']:.0%} "
        f"timeout={variant['timeout_rate']:.0%} submit={variant['submit_rate']:.0%}"
    )
    gate = _go({}, variant)

    lines = [
        "# Full Rule-BC + evidence-gate PPO",
        "",
        "> In-process: Rule-BC (pad=6) → MaskablePPO 30k with **evidence diagnose-gate**.",
        "> Named variant. Original reward after unlock. Avoids cross-process action-id drift.",
        f"> Smoke: {smoke}",
        "",
        f"- Demos: **{collect['ok']}**, BC epochs {cfg['bc_epochs']}, last loss {losses[-1]:.4f}",
        f"- PPO timesteps: {cfg['timesteps']}, pad={pad}",
        "",
        "## Exact / behavior",
        "",
        "| Stage | Exact | Shortcut | Timeout | Submit | AvgSteps |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Full Rule-BC (gate OFF) | {_exact(seed_off):.0%} | {seed_off['shortcut']['shortcut_rate']:.0%} | "
        f"{seed_off['timeout_rate']:.0%} | {seed_off['submit_rate']:.0%} | "
        f"{seed_off['stats']['avg_steps_all']:.1f} |",
        f"| Full Rule-BC zero-shot (gate ON) | {_exact(seed_on):.0%} | {seed_on['shortcut']['shortcut_rate']:.0%} | "
        f"{seed_on['timeout_rate']:.0%} | {seed_on['submit_rate']:.0%} | "
        f"{seed_on['stats']['avg_steps_all']:.1f} |",
        f"| Full Rule-BC + Gate PPO | {_exact(variant):.0%} | {variant['shortcut']['shortcut_rate']:.0%} | "
        f"{variant['timeout_rate']:.0%} | {variant['submit_rate']:.0%} | "
        f"{variant['stats']['avg_steps_all']:.1f} |",
        f"| Short-seed + Gate PPO (prior) | ~2% | 0% | 50% | 50% | 10.1 |",
        f"| Gate-only PPO (prior) | ~0% | 0% | **76%** | ~22% | — |",
        "",
        "## Per-fault after Gate PPO",
        "",
        f"- Device {_pf(variant, 'device_failure')}",
        f"- Perf {_pf(variant, 'performance_degradation')}",
        f"- Misconfig {_pf(variant, 'misconfiguration')}",
        f"- Link {_pf(variant, 'link_failure')}",
        "",
        "## Zero-shot under gate (before PPO)",
        "",
        f"- Device {_pf(seed_on, 'device_failure')}",
        f"- Perf {_pf(seed_on, 'performance_degradation')}",
        f"- Misconfig {_pf(seed_on, 'misconfiguration')}",
        f"- Link {_pf(seed_on, 'link_failure')}",
        "",
        f"- Train exact∧cons\\|tool: {train_sum.get('p_exact_cons_given_tool', 0):.1%}",
        f"- Decision: **{gate['decision']}**",
        f"- {gate['rule']}",
        "",
        "Run: `python -m src.run_bc_rule_gate_ppo`",
        "",
    ]
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "bc_rule_gate_ppo_poc.md").write_text("\n".join(lines), encoding="utf-8")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("records",)}
        return o

    payload = {
        "smoke": smoke,
        "collect": collect,
        "bc_losses": losses,
        "seed_gate_off": _clean(seed_off),
        "seed_gate_on_zeroshot": _clean(seed_on),
        "variant": _clean(variant),
        "train": train_sum,
        "gate": gate,
        "pad": pad,
    }
    (results / "bc_rule_gate_ppo_poc.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"decision={gate['decision']} exact={_exact(variant):.0%} timeout={variant['timeout_rate']:.0%}")
    print("Wrote", reports / "bc_rule_gate_ppo_poc.md")


if __name__ == "__main__":
    main()
