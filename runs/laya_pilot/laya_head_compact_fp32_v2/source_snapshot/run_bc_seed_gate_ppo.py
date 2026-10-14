"""Short Rule-BC seed + evidence diagnose-gate + MaskablePPO.

Named variant. Gate blocks diagnosis until fault-like obs bits.
Seed demos: GT-free Rule trajectories (gate OFF while cloning).
PPO: same original reward, gate ON, 30k.
"""
from __future__ import annotations

import json
from typing import Dict, List

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_trace import aggregate_behavior, run_episode_with_policy
from .diagnosis_decomposition import aggregate_decomposition
from .env_utils import make_masked_base_env, make_masked_eval_env, metrics_env
from .evidence_diagnose_gate import smoke_test
from .metrics_utils import summarize_evaluator
from .pbrs import find_pbrs
from .protocol import PROJECT_ROOT, load_protocol
from .run_ablation_nl_poc import _per_fault, _row_consistent, _selector
from .run_diagnosis_gate_poc import shortcut_rate
from .run_evidence_diagnose_gate import _go
from .run_star_scale_poc import _env_kwargs, _exact, _pf, bc_train, collect_rule_demos, evaluate_model
from .run_symmetric_reward_poc import set_seeds
from .train_episode_log import find_train_log, summarize_rows


def _evaluate_pad_gate(
    model: MaskablePPO,
    protocol: dict,
    *,
    pad: int,
    n: int,
    seed_base: int,
    det: bool,
    gate: bool,
    n_devices: int = 6,
) -> Dict:
    """Same metrics as evidence-gate eval, with star pad + optional diagnose gate."""
    cfg = protocol.get("bc_rule_gate_ppo_poc") or protocol.get("bc_seed_gate_ppo_poc") or {}
    topologies = cfg.get("topology_types") or protocol.get("train_topology_types") or ["star"]
    if isinstance(topologies, str):
        topologies = [topologies]
    kw = _env_kwargs(protocol, n_devices, pad)
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    evaluator = CompetitionEvaluator()
    records: List = []
    summaries: List = []
    rows: List[Dict] = []
    max_steps = kw["max_episode_steps"]
    for i in range(n):
        fault = faults[i % len(faults)]
        env = make_masked_eval_env(
            topologies,
            fault_types=[fault],
            protocol=protocol,
            evidence_diagnose_gate=gate,
            pbrs=True,
            pbrs_apply_shaping=False,
            **kw,
        )
        recs, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed_base + i,
            agent="bc_seed_gate" if gate else "bc_seed",
            choose_action=_selector(model, deterministic=det),
        )
        m = metrics_env(env).last_episode_metrics
        if m:
            evaluator.add_episode_metrics(m)
        tracker = find_pbrs(env)
        bits = tracker.bits_before_diagnosis or tracker.bits.snapshot()
        phi = tracker.phi_before_diagnosis
        if phi is None:
            phi = tracker.bits.phi
        rows.append(
            {
                "fault_type": summary.fault_type,
                "fault_location": summary.fault_location,
                "success": summary.success,
                "predicted_type": summary.predicted_type,
                "predicted_location": summary.predicted_location,
                "phi_before_dx": int(phi),
                "type_bit": int(bits.get("type_bit", 0)),
                "loc_bit": int(bits.get("loc_bit", 0)),
                "steps": summary.steps,
                "submitted": summary.predicted_type is not None,
            }
        )
        records.extend(recs)
        summaries.append(summary)
        env.close()
    timeout = sum(1 for s in summaries if (not s.success) and s.steps >= max_steps)
    exact_cons = [r for r in rows if r["success"] and _row_consistent(r)]
    return {
        "stats": summarize_evaluator(evaluator),
        "behavior": aggregate_behavior(summaries, records),
        "decomposition": aggregate_decomposition(summaries),
        "per_fault_dsr": aggregate_behavior(summaries, records).get("fault_type_dsr", {}),
        "per_fault": _per_fault(rows, summaries),
        "shortcut": shortcut_rate(summaries, records),
        "timeout_rate": timeout / max(1, n),
        "submit_rate": sum(1 for r in rows if r["submitted"]) / max(1, n),
        "exact_consistent_n": len(exact_cons),
        "n": n,
        "max_steps": max_steps,
    }


def main() -> None:
    protocol = load_protocol()
    cfg = protocol.setdefault(
        "bc_seed_gate_ppo_poc",
        {
            "topology_types": ["star"],
            "seed_demos": 400,
            "seed_bc_epochs": 15,
            "timesteps": 30000,
            "policy_seed": 1,
            "eval_episodes": 100,
            "eval_seed_base": 111000,
        },
    )
    # helpers that read star_scale_poc topology
    protocol.setdefault("star_scale_poc", {"topology_types": ["star"]})
    protocol["star_scale_poc"]["topology_types"] = cfg["topology_types"]

    smoke = smoke_test()
    print("smoke", smoke)
    if not all(smoke.values()):
        raise SystemExit(f"smoke failed: {smoke}")

    set_seeds(int(cfg["policy_seed"]))
    topologies = cfg["topology_types"]

    # --- 1) Short Rule BC (no gate, evidence obs, pad=6 = fixed star action ids) ---
    print(f"=== Seed Rule-BC demos={cfg['seed_demos']} epochs={cfg['seed_bc_epochs']} ===")
    pad = 6
    protocol["min_devices"] = 6
    protocol["max_devices"] = 6
    obs, acts, masks, collect = collect_rule_demos(
        protocol,
        n_devices=6,
        pad=pad,
        n_ok=int(cfg["seed_demos"]),
        seed0=60_000,
    )
    print("collect", collect)
    if collect["ok"] < 100:
        raise SystemExit(f"too few demos: {collect}")

    env_bc = make_masked_base_env(
        topologies,
        protocol=protocol,
        evidence_obs=True,
        evidence_diagnose_gate=False,
        min_devices=6,
        max_devices=6,
        star_pad_max=pad,
    )
    check_env(env_bc, warn=True)
    model = MaskablePPO("MultiInputPolicy", env_bc, verbose=0, seed=int(cfg["policy_seed"]))
    losses = bc_train(
        model,
        obs,
        acts,
        masks,
        epochs=int(cfg["seed_bc_epochs"]),
        batch=64,
        lr=3e-4,
    )
    seed_ckpt = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_bc_seed_pre_gate.zip"
    seed_ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(seed_ckpt.with_suffix("")))
    env_bc.close()

    # Seed-only greedy eval (no gate)
    print("Eval seed BC (gate OFF)...")
    seed_eval = evaluate_model(
        model,
        protocol,
        n_devices=6,
        pad=pad,
        n=int(cfg["eval_episodes"]),
        seed_base=int(cfg["eval_seed_base"]),
        det=True,
    )
    print(f"  seed exact={_exact(seed_eval):.0%} shortcut={seed_eval['shortcut']['shortcut_rate']:.0%}")

    # --- 2) PPO with evidence gate (same pad) ---
    log_path = PROJECT_ROOT / "results" / "train_episode_log" / "bc_seed_gate_ppo_star.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = make_masked_base_env(
        topologies,
        protocol=protocol,
        evidence_obs=True,
        evidence_diagnose_gate=True,
        episode_log_path=log_path,
        min_devices=6,
        max_devices=6,
        star_pad_max=pad,
    )
    check_env(env, warn=True)
    model = MaskablePPO.load(str(seed_ckpt), env=env)
    print(f"=== PPO + evidence gate {cfg['timesteps']} steps ===")
    model.learn(total_timesteps=int(cfg["timesteps"]))
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_bc_seed_gate_ppo_star.zip"
    model.save(str(ckpt.with_suffix("")))
    logger = find_train_log(env)
    train_sum = summarize_rows(logger.rows) if logger else {}
    env.close()

    n = int(cfg["eval_episodes"])
    seed_base = int(cfg["eval_seed_base"])
    print("Eval after PPO (gate ON)...")
    # _evaluate from evidence gate does not pass star_pad — use local eval with pad+gate
    variant = _evaluate_pad_gate(
        model, protocol, pad=pad, n=n, seed_base=seed_base + 100, det=True, gate=True
    )
    print(
        f"  exact={_exact(variant):.0%} shortcut={variant['shortcut']['shortcut_rate']:.0%} "
        f"timeout={variant['timeout_rate']:.0%} submit={variant['submit_rate']:.0%}"
    )
    gate = _go({}, variant)

    # Prior refs
    prior_gate = {"exact": 0.0, "shortcut": 0.0, "timeout": 0.76, "submit": 0.22}
    prior_bc_rule = {"exact": 0.99, "shortcut": 0.0}

    lines = [
        "# Rule-BC seed + evidence-gate PPO",
        "",
        "> 1) Short GT-free Rule BC (no gate).",
        "> 2) MaskablePPO 30k with **evidence diagnose-gate** (no submit without fault-like obs).",
        "> Named variant. Original reward after unlock.",
        f"> Smoke: {smoke}",
        "",
        f"- Seed demos: **{collect['ok']}**, BC epochs {cfg['seed_bc_epochs']}, last loss {losses[-1]:.4f}",
        f"- PPO timesteps: {cfg['timesteps']}",
        "",
        "## Exact / behavior",
        "",
        "| Stage | Exact | Shortcut | Timeout | Submit | AvgSteps |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Seed BC (gate OFF) | {_exact(seed_eval):.0%} | {seed_eval['shortcut']['shortcut_rate']:.0%} | "
        f"{seed_eval['timeout_rate']:.0%} | {seed_eval['submit_rate']:.0%} | "
        f"{seed_eval['stats']['avg_steps_all']:.1f} |",
        f"| Seed+Gate PPO (gate ON) | {_exact(variant):.0%} | {variant['shortcut']['shortcut_rate']:.0%} | "
        f"{variant['timeout_rate']:.0%} | {variant['submit_rate']:.0%} | "
        f"{variant['stats']['avg_steps_all']:.1f} |",
        f"| Gate-only PPO (prior) | ~0% | 0% | **76%** | ~22% | — |",
        f"| Full Rule-BC (prior) | 99% | 0% | 0% | 100% | 6.0 |",
        "",
        "## Per-fault after Seed+Gate PPO",
        "",
        f"- Device {_pf(variant, 'device_failure')}",
        f"- Perf {_pf(variant, 'performance_degradation')}",
        f"- Misconfig {_pf(variant, 'misconfiguration')}",
        f"- Link {_pf(variant, 'link_failure')}",
        "",
        f"- Train exact∧cons\\|tool: {train_sum.get('p_exact_cons_given_tool', 0):.1%}",
        f"- Decision: **{gate['decision']}**",
        f"- {gate['rule']}",
        "",
        "Run: `python -m src.run_bc_seed_gate_ppo`",
        "",
    ]
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "bc_seed_gate_ppo_poc.md").write_text("\n".join(lines), encoding="utf-8")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("records",)}
        return o

    payload = {
        "smoke": smoke,
        "collect": collect,
        "bc_losses": losses,
        "seed_eval": _clean(seed_eval),
        "variant": _clean(variant),
        "train": train_sum,
        "gate": gate,
        "prior_gate_only": prior_gate,
        "prior_bc_rule": prior_bc_rule,
    }
    (results / "bc_seed_gate_ppo_poc.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"decision={gate['decision']} exact={_exact(variant):.0%} timeout={variant['timeout_rate']:.0%}")
    print("Wrote", reports / "bc_seed_gate_ppo_poc.md")


if __name__ == "__main__":
    main()
