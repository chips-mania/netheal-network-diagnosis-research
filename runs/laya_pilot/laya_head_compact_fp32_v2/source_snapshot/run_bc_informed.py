"""BC on scripted informed trajectories (scan → evidence tool → exact dx).

Named variant. Expert uses GT only to generate demos. Eval has no GT in the policy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from .action_mask import action_masks_for_env
from .action_trace import _diagnosis_correct, _parse_action_spec
from .env_utils import make_masked_base_env
from .protocol import PROJECT_ROOT, load_protocol
from .run_evidence_diagnose_gate import _evaluate, _go
from .run_symmetric_reward_poc import set_seeds
from .scripted_expert import expert_action
from .action_mask import base_env


def _collect_demos(protocol, n_ok: int) -> Tuple[List[dict], List[int], List[np.ndarray], Dict]:
    cfg = protocol["bc_informed_poc"]
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
    while ok < n_ok and tries < n_ok * 5:
        tries += 1
        obs, info = env.reset(seed=10_000 + tries)
        root = base_env(env)
        gt = info.get("ground_truth_fault") or {}
        ft = gt.get("type", "unknown")
        success = False
        ep_obs, ep_act, ep_mask = [], [], []
        for t in range(3):
            mask = action_masks_for_env(env)
            try:
                action = expert_action(env, t)
            except RuntimeError:
                success = False
                break
            ep_obs.append({k: np.array(v, copy=True) for k, v in obs.items()})
            ep_act.append(int(action))
            ep_mask.append(np.array(mask, copy=True))
            obs, reward, term, trunc, info = env.step(action)
            spec = _parse_action_spec(info, root, action)
            if spec is not None and _diagnosis_correct(info, spec) is True:
                success = True
            if term or trunc:
                break
        if success:
            ok += 1
            per[ft] = per.get(ft, 0) + 1
            obs_list.extend(ep_obs)
            act_list.extend(ep_act)
            mask_list.extend(ep_mask)
    env.close()
    stats = {"ok": ok, "tries": tries, "per_fault_ok": per, "n_transitions": len(act_list)}
    return obs_list, act_list, mask_list, stats


def _bc_train(model: MaskablePPO, obs_list, act_list, mask_list, *, epochs: int, batch: int) -> List[float]:
    opt = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    n = len(act_list)
    losses = []
    device = model.device
    for ep in range(epochs):
        idx = np.random.permutation(n)
        ep_loss = []
        for start in range(0, n, batch):
            sel = idx[start : start + batch]
            if len(sel) < 2:
                continue
            keys = obs_list[0].keys()
            batch_obs = {k: np.stack([obs_list[i][k] for i in sel], axis=0) for k in keys}
            acts = torch.as_tensor([act_list[i] for i in sel], device=device, dtype=torch.int64)
            masks = torch.as_tensor(np.stack([mask_list[i] for i in sel], axis=0), device=device)
            obs_t, _ = model.policy.obs_to_tensor(batch_obs)
            dist = model.policy.get_distribution(obs_t, action_masks=masks)
            loss = -dist.log_prob(acts).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss.append(float(loss.item()))
        losses.append(float(np.mean(ep_loss)) if ep_loss else 0.0)
        if (ep + 1) % 10 == 0:
            print(f"bc epoch {ep+1}/{epochs} loss={losses[-1]:.4f}")
    return losses


def main() -> None:
    protocol = load_protocol()
    if "bc_informed_poc" not in protocol:
        protocol["bc_informed_poc"] = {
            "topology_types": ["star"],
            "n_demos": 2000,
            "bc_epochs": 40,
            "batch_size": 64,
            "eval_episodes": 100,
            "eval_seed_base": 105000,
            "policy_seed": 1,
        }
    cfg = protocol["bc_informed_poc"]
    set_seeds(cfg["policy_seed"])

    print(f"Collecting {cfg['n_demos']} informed demos...")
    obs_list, act_list, mask_list, collect = _collect_demos(protocol, cfg["n_demos"])
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
    losses = _bc_train(model, obs_list, act_list, mask_list, epochs=cfg["bc_epochs"], batch=cfg["batch_size"])
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_bc_informed_star.zip"
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
        return f"{b.get('dsr', 0):.0%} ({b.get('exact_n', 0)}/{b.get('n', 0)}; cons {b.get('exact_consistent_n', 0)})"

    lines = [
        "# BC on informed trajectories",
        "",
        "> Expert (GT only for demo gen): scan → evidence tool → exact diagnosis.",
        "> Policy never sees GT. Evidence obs ON, diagnose-gate OFF (clone the 3-step path itself).",
        "",
        f"- Demos kept: **{collect['ok']}** / tries {collect['tries']} ({collect['n_transitions']} transitions)",
        f"- Per-fault demos: {collect['per_fault_ok']}",
        f"- Last BC loss: {losses[-1]:.4f}",
        "",
        "| Metric | BC-informed |",
        "|---|---:|",
        f"| Exact DSR | {variant['decomposition']['exact_diagnosis_accuracy']:.0%} |",
        f"| Shortcut | {variant['shortcut']['shortcut_rate']:.0%} |",
        f"| AvgSteps | {variant['stats']['avg_steps_all']:.1f} |",
        f"| Timeout | {variant['timeout_rate']:.0%} |",
        f"| Device | {_pf('device_failure')} |",
        f"| Perf | {_pf('performance_degradation')} |",
        f"| Misconfig | {_pf('misconfiguration')} |",
        f"| Link | {_pf('link_failure')} |",
        "",
        f"- Decision: **{gate['decision']}** (same hard-fault rule as gate)",
        "",
    ]
    reports = PROJECT_ROOT / "reports"
    results = PROJECT_ROOT / "results"
    (reports / "bc_informed_poc.md").write_text("\n".join(lines), encoding="utf-8")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("records",)}
        return o

    (results / "bc_informed_poc.json").write_text(
        json.dumps({"collect": collect, "bc_losses": losses, "variant": _clean(variant), "gate": gate}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"decision={gate['decision']} exact={variant['decomposition']['exact_diagnosis_accuracy']:.0%}")
    print("Wrote", reports / "bc_informed_poc.md")


if __name__ == "__main__":
    main()
