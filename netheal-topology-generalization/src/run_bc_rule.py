"""BC on GT-free Rule trajectories (disambiguation tree), not GT 3-step expert.

Named variant. Demos from RuleAgent (no GT in action choice). Eval has no GT in policy.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from .action_mask import action_masks_for_env, base_env
from .action_trace import _diagnosis_correct, _parse_action_spec
from .env_utils import make_masked_base_env
from .protocol import PROJECT_ROOT, load_protocol
from .rule_agent import RuleAgent
from .run_bc_informed import _bc_train
from .run_evidence_diagnose_gate import _evaluate, _go
from .run_symmetric_reward_poc import set_seeds


def _collect_rule_demos(protocol, n_ok: int) -> Tuple[List[dict], List[int], List[np.ndarray], Dict]:
    cfg = protocol["bc_rule_poc"]
    max_steps = int(protocol["max_episode_steps"])
    env = make_masked_base_env(
        cfg["topology_types"],
        protocol=protocol,
        evidence_obs=True,
    )
    obs_list: List[dict] = []
    act_list: List[int] = []
    mask_list: List[np.ndarray] = []
    ok = 0
    tries = 0
    per: Dict[str, int] = {}
    lengths: List[int] = []
    while ok < n_ok and tries < n_ok * 5:
        tries += 1
        agent = RuleAgent()
        obs, info = env.reset(seed=20_000 + tries)
        root = base_env(env)
        gt = info.get("ground_truth_fault") or {}
        ft = gt.get("type", "unknown")
        success = False
        ep_obs, ep_act, ep_mask = [], [], []
        for t in range(1, max_steps + 1):
            mask = action_masks_for_env(env)
            try:
                action = int(agent.choose(env, obs, t, []))
            except RuntimeError:
                success = False
                break
            ep_obs.append({k: np.array(v, copy=True) for k, v in obs.items()})
            ep_act.append(action)
            ep_mask.append(np.array(mask, copy=True))
            obs, reward, term, trunc, info = env.step(action)
            agent.observe_step(info)
            spec = _parse_action_spec(info, root, action)
            if spec is not None and _diagnosis_correct(info, spec) is True:
                success = True
            if term or trunc:
                break
        if success:
            ok += 1
            per[ft] = per.get(ft, 0) + 1
            lengths.append(len(ep_act))
            obs_list.extend(ep_obs)
            act_list.extend(ep_act)
            mask_list.extend(ep_mask)
    env.close()
    stats = {
        "ok": ok,
        "tries": tries,
        "per_fault_ok": per,
        "n_transitions": len(act_list),
        "avg_demo_len": float(np.mean(lengths)) if lengths else 0.0,
    }
    return obs_list, act_list, mask_list, stats


def main() -> None:
    protocol = load_protocol()
    if "bc_rule_poc" not in protocol:
        protocol["bc_rule_poc"] = {
            "topology_types": ["star"],
            "n_demos": 2000,
            "bc_epochs": 40,
            "batch_size": 64,
            "eval_episodes": 100,
            "eval_seed_base": 107000,
            "policy_seed": 1,
        }
    cfg = protocol["bc_rule_poc"]
    set_seeds(cfg["policy_seed"])

    print(f"Collecting {cfg['n_demos']} rule demos...")
    obs_list, act_list, mask_list, collect = _collect_rule_demos(protocol, cfg["n_demos"])
    print("collect", collect)
    if collect["ok"] < 100:
        raise SystemExit(f"too few demos: {collect}")

    env = make_masked_base_env(
        cfg["topology_types"],
        protocol=protocol,
        evidence_obs=True,
    )
    check_env(env, warn=True)
    model = MaskablePPO("MultiInputPolicy", env, verbose=0, seed=cfg["policy_seed"])
    print("BC train...")
    losses = _bc_train(
        model, obs_list, act_list, mask_list, epochs=cfg["bc_epochs"], batch=cfg["batch_size"]
    )
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_bc_rule_star.zip"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt.with_suffix("")))
    env.close()

    print("Eval det...")
    variant = _evaluate(
        model, protocol, gate=False, n=cfg["eval_episodes"], seed_base=cfg["eval_seed_base"], det=True
    )
    gate = _go({}, variant)

    def _pf(ft):
        b = variant["per_fault"].get(ft, {})
        return (
            f"{b.get('dsr', 0):.0%} ({b.get('exact_n', 0)}/{b.get('n', 0)}; "
            f"cons {b.get('exact_consistent_n', 0)})"
        )

    lines = [
        "# BC on Rule trajectories",
        "",
        "> Demos from GT-free RuleAgent (scan → disambiguate tools → diagnose).",
        "> Not the GT 3-step expert. Policy never sees GT. Evidence obs ON, gate OFF.",
        "",
        f"- Demos kept: **{collect['ok']}** / tries {collect['tries']} "
        f"({collect['n_transitions']} transitions, avg len {collect['avg_demo_len']:.1f})",
        f"- Per-fault demos: {collect['per_fault_ok']}",
        f"- Last BC loss: {losses[-1]:.4f}",
        "",
        "| Metric | BC-rule | (ref BC-informed) |",
        "|---|---:|---:|",
        f"| Exact DSR | {variant['decomposition']['exact_diagnosis_accuracy']:.0%} | 58% |",
        f"| Shortcut | {variant['shortcut']['shortcut_rate']:.0%} | 0% |",
        f"| AvgSteps | {variant['stats']['avg_steps_all']:.1f} | 3.3 |",
        f"| Timeout | {variant['timeout_rate']:.0%} | 0% |",
        f"| Device | {_pf('device_failure')} | 8% |",
        f"| Perf | {_pf('performance_degradation')} | 28% |",
        f"| Misconfig | {_pf('misconfiguration')} | 96% |",
        f"| Link | {_pf('link_failure')} | 100% |",
        "",
        f"- Decision: **{gate['decision']}** (same hard-fault rule as gate)",
        "",
    ]
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "bc_rule_poc.md").write_text("\n".join(lines), encoding="utf-8")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("records",)}
        return o

    (results / "bc_rule_poc.json").write_text(
        json.dumps(
            {"collect": collect, "bc_losses": losses, "variant": _clean(variant), "gate": gate},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        f"decision={gate['decision']} exact={variant['decomposition']['exact_diagnosis_accuracy']:.0%}"
    )
    print("Wrote", reports / "bc_rule_poc.md")


if __name__ == "__main__":
    main()
