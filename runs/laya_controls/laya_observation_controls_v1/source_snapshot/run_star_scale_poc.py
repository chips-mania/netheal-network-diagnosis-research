"""Star device-count scale + pad12 BC-rule base + few-shot FT.

Named variant. Fixed action/obs pad so N=6 weights load on N=8/10.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_mask import action_masks_for_env, base_env
from .action_trace import (
    aggregate_behavior,
    run_episode_with_policy,
    _diagnosis_correct,
    _parse_action_spec,
)
from .diagnosis_decomposition import aggregate_decomposition
from .env_utils import make_masked_base_env, make_masked_eval_env, metrics_env
from .metrics_utils import summarize_evaluator
from .pbrs import find_pbrs
from .protocol import PROJECT_ROOT, load_protocol
from .rule_agent import RuleAgent, run_episode_rule
from .run_ablation_nl_poc import _per_fault, _row_consistent, _selector
from .run_bc_informed import _bc_train
from .run_diagnosis_gate_poc import shortcut_rate
from .run_evidence_diagnose_gate import _go
from .run_symmetric_reward_poc import set_seeds
from .star_pad import protocol_for_star_n, steps_budget


def _env_kwargs(protocol: dict, n: int, pad: int) -> dict:
    p = protocol_for_star_n(protocol, n, pad)
    return {
        "min_devices": n,
        "max_devices": n,
        "max_episode_steps": p["max_episode_steps"],
        "star_pad_max": pad,
        "evidence_obs": True,
    }


def collect_rule_demos(
    protocol: dict,
    *,
    n_devices: int,
    pad: int,
    n_ok: int,
    seed0: int,
) -> Tuple[List[dict], List[int], List[np.ndarray], Dict]:
    kw = _env_kwargs(protocol, n_devices, pad)
    max_steps = kw["max_episode_steps"]
    env = make_masked_base_env(protocol["star_scale_poc"]["topology_types"], protocol=protocol, **kw)
    obs_list: List[dict] = []
    act_list: List[int] = []
    mask_list: List[np.ndarray] = []
    ok = 0
    tries = 0
    per: Dict[str, int] = {}
    lengths: List[int] = []
    while ok < n_ok and tries < n_ok * 8:
        tries += 1
        agent = RuleAgent()
        obs, info = env.reset(seed=seed0 + tries)
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
    return (
        obs_list,
        act_list,
        mask_list,
        {
            "ok": ok,
            "tries": tries,
            "per_fault_ok": per,
            "n_transitions": len(act_list),
            "avg_demo_len": float(np.mean(lengths)) if lengths else 0.0,
            "n_devices": n_devices,
            "pad": pad,
            "max_steps": max_steps,
        },
    )


def bc_train(
    model: MaskablePPO,
    obs_list,
    act_list,
    mask_list,
    *,
    epochs: int,
    batch: int,
    lr: float = 3e-4,
) -> List[float]:
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
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
        if (ep + 1) % max(1, epochs // 4) == 0 or ep + 1 == epochs:
            print(f"  bc epoch {ep+1}/{epochs} loss={losses[-1]:.4f}")
    return losses


def evaluate_rule(protocol: dict, *, n_devices: int, pad: int, n: int, seed_base: int) -> Dict:
    cfg = protocol["star_scale_poc"]
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
            cfg["topology_types"],
            fault_types=[fault],
            protocol=protocol,
            pbrs=True,
            pbrs_apply_shaping=False,
            **kw,
        )
        recs, summary = run_episode_rule(env, episode_id=i, seed=seed_base + i)
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
    return {
        "n_devices": n_devices,
        "stats": summarize_evaluator(evaluator),
        "behavior": aggregate_behavior(summaries, records),
        "decomposition": aggregate_decomposition(summaries),
        "per_fault": _per_fault(rows, summaries),
        "shortcut": shortcut_rate(summaries, records),
        "timeout_rate": timeout / max(1, n),
        "submit_rate": sum(1 for r in rows if r["submitted"]) / max(1, n),
        "exact_consistent_n": sum(1 for r in rows if r["success"] and _row_consistent(r)),
        "n": n,
        "max_steps": max_steps,
    }


def evaluate_model(
    model: MaskablePPO,
    protocol: dict,
    *,
    n_devices: int,
    pad: int,
    n: int,
    seed_base: int,
    det: bool = True,
) -> Dict:
    cfg = protocol["star_scale_poc"]
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
            cfg["topology_types"],
            fault_types=[fault],
            protocol=protocol,
            pbrs=True,
            pbrs_apply_shaping=False,
            **kw,
        )
        recs, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed_base + i,
            agent=f"bc_pad_n{n_devices}",
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
        "n_devices": n_devices,
        "stats": summarize_evaluator(evaluator),
        "behavior": aggregate_behavior(summaries, records),
        "decomposition": aggregate_decomposition(summaries),
        "per_fault": _per_fault(rows, summaries),
        "shortcut": shortcut_rate(summaries, records),
        "timeout_rate": timeout / max(1, n),
        "submit_rate": sum(1 for r in rows if r["submitted"]) / max(1, n),
        "exact_consistent_n": len(exact_cons),
        "n": n,
        "max_steps": max_steps,
    }


def _pf(variant: Dict, ft: str) -> str:
    b = variant["per_fault"].get(ft, {})
    return (
        f"{b.get('dsr', 0):.0%} ({b.get('exact_n', 0)}/{b.get('n', 0)}; "
        f"cons {b.get('exact_consistent_n', 0)})"
    )


def _exact(v: Dict) -> float:
    return float(v["decomposition"]["exact_diagnosis_accuracy"])


def main() -> None:
    protocol = load_protocol()
    cfg = protocol["star_scale_poc"]
    pad = int(cfg["pad_max_devices"])
    base_n = int(cfg["base_n"])
    n_eval = int(cfg["eval_episodes"])
    seed = int(cfg["policy_seed"])
    set_seeds(seed)
    topologies = cfg["topology_types"]
    results: Dict = {"pad": pad, "base_n": base_n}

    # --- 1) Base BC at N=6 under pad ---
    print(f"=== Base BC-rule N={base_n} pad={pad} demos={cfg['base_demos']} ===")
    obs, acts, masks, collect = collect_rule_demos(
        protocol,
        n_devices=base_n,
        pad=pad,
        n_ok=int(cfg["base_demos"]),
        seed0=30_000,
    )
    print("collect", collect)
    if collect["ok"] < 100:
        raise SystemExit(f"too few base demos: {collect}")

    env = make_masked_base_env(topologies, protocol=protocol, **_env_kwargs(protocol, base_n, pad))
    check_env(env, warn=True)
    model = MaskablePPO("MultiInputPolicy", env, verbose=0, seed=seed)
    print("BC train base...")
    losses = bc_train(
        model,
        obs,
        acts,
        masks,
        epochs=int(cfg["base_bc_epochs"]),
        batch=int(cfg["batch_size"]),
        lr=3e-4,
    )
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_bc_rule_pad12.zip"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt.with_suffix("")))
    env.close()
    results["base_collect"] = collect
    results["base_bc_losses"] = losses

    print(f"Eval base on N={base_n}...")
    base_eval = evaluate_model(
        model,
        protocol,
        n_devices=base_n,
        pad=pad,
        n=n_eval,
        seed_base=int(cfg["eval_seed_base"]),
    )
    results["base_eval_n6"] = base_eval
    print(f"  exact={_exact(base_eval):.0%} shortcut={base_eval['shortcut']['shortcut_rate']:.0%}")

    # --- 2) Rule ceiling N=8,10 ---
    rule_by_n: Dict[str, Dict] = {}
    for n_dev in cfg["n_list"]:
        n_dev = int(n_dev)
        print(f"=== Rule ceiling N={n_dev} ===")
        rule_by_n[str(n_dev)] = evaluate_rule(
            protocol,
            n_devices=n_dev,
            pad=pad,
            n=n_eval,
            seed_base=int(cfg["eval_seed_base"]) + 1000 * n_dev,
        )
        r = rule_by_n[str(n_dev)]
        print(f"  exact={_exact(r):.0%} steps={r['stats']['avg_steps_all']:.1f} timeout={r['timeout_rate']:.0%}")
    results["rule"] = rule_by_n

    # --- 3) Zero-shot pad12 on N=8,10 ---
    zero_by_n: Dict[str, Dict] = {}
    for n_dev in cfg["n_list"]:
        n_dev = int(n_dev)
        if n_dev == base_n:
            zero_by_n[str(n_dev)] = base_eval
            continue
        print(f"=== Zero-shot pad12 on N={n_dev} ===")
        zero_by_n[str(n_dev)] = evaluate_model(
            model,
            protocol,
            n_devices=n_dev,
            pad=pad,
            n=n_eval,
            seed_base=int(cfg["eval_seed_base"]) + 2000 * n_dev,
        )
        z = zero_by_n[str(n_dev)]
        print(f"  exact={_exact(z):.0%} shortcut={z['shortcut']['shortcut_rate']:.0%}")
    results["zeroshot"] = zero_by_n

    # --- 4) Few-shot FT N=8 then N=10 ---
    ft_by_n: Dict[str, Dict] = {}
    ft_collect: Dict[str, Dict] = {}
    ft_losses: Dict[str, List[float]] = {}
    for n_dev in [8, 10]:
        print(f"=== Few-shot BC-FT N={n_dev} demos={cfg['ft_demos']} epochs={cfg['ft_bc_epochs']} ===")
        obs_f, acts_f, masks_f, col_f = collect_rule_demos(
            protocol,
            n_devices=n_dev,
            pad=pad,
            n_ok=int(cfg["ft_demos"]),
            seed0=40_000 + 1000 * n_dev,
        )
        print("  collect", col_f)
        ft_collect[str(n_dev)] = col_f
        if col_f["ok"] < 50:
            print("  WARN: few demos, skipping FT")
            continue
        losses_f = bc_train(
            model,
            obs_f,
            acts_f,
            masks_f,
            epochs=int(cfg["ft_bc_epochs"]),
            batch=int(cfg["batch_size"]),
            lr=float(cfg["ft_lr"]),
        )
        ft_losses[str(n_dev)] = losses_f
        ft_path = PROJECT_ROOT / "models" / "checkpoints" / f"masked_ppo_bc_rule_pad12_ft{n_dev}.zip"
        model.save(str(ft_path.with_suffix("")))
        print(f"Eval FT on N={n_dev}...")
        ft_by_n[str(n_dev)] = evaluate_model(
            model,
            protocol,
            n_devices=n_dev,
            pad=pad,
            n=n_eval,
            seed_base=int(cfg["eval_seed_base"]) + 3000 * n_dev,
        )
        ftv = ft_by_n[str(n_dev)]
        print(
            f"  exact={_exact(ftv):.0%} Device={_pf(ftv, 'device_failure')} "
            f"Perf={_pf(ftv, 'performance_degradation')}"
        )
        # also re-check N=6 retention after FT chain
        print(f"  Retention eval N={base_n} after FT{n_dev}...")
        ft_by_n[f"retain6_after_{n_dev}"] = evaluate_model(
            model,
            protocol,
            n_devices=base_n,
            pad=pad,
            n=n_eval,
            seed_base=int(cfg["eval_seed_base"]) + 4000 * n_dev,
        )
        print(f"  retain6 exact={_exact(ft_by_n[f'retain6_after_{n_dev}']):.0%}")

    results["ft"] = ft_by_n
    results["ft_collect"] = ft_collect
    results["ft_losses"] = ft_losses

    # --- Report ---
    lines = [
        "# Star scale + pad12 fine-tune",
        "",
        f"> Obs/action padded to **max_devices={pad}** (canonical `center`/`edge_*`).",
        f"> Base: Rule-BC on Star N={base_n}. FT: few-shot Rule-BC on N=8 then N=10.",
        "",
        f"- Base demos: **{collect['ok']}** (avg len {collect['avg_demo_len']:.1f}), "
        f"last BC loss {losses[-1]:.4f}",
        f"- FT demos/epochs: {cfg['ft_demos']} / {cfg['ft_bc_epochs']} (lr={cfg['ft_lr']})",
        f"- ckpt: `models/checkpoints/masked_ppo_bc_rule_pad12.zip`",
        "",
        "## Summary",
        "",
        "| N | Rule Exact | Zero-shot Exact | FT Exact | Shortcut (FT or zero) | AvgSteps |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n_dev in cfg["n_list"]:
        ns = str(n_dev)
        rule_e = _exact(rule_by_n[ns])
        zero_e = _exact(zero_by_n[ns])
        if ns in ft_by_n:
            ft_e = _exact(ft_by_n[ns])
            sc = ft_by_n[ns]["shortcut"]["shortcut_rate"]
            steps = ft_by_n[ns]["stats"]["avg_steps_all"]
        else:
            ft_e = float("nan")
            sc = zero_by_n[ns]["shortcut"]["shortcut_rate"]
            steps = zero_by_n[ns]["stats"]["avg_steps_all"]
        ft_s = f"{ft_e:.0%}" if ns in ft_by_n else "—"
        lines.append(
            f"| {n_dev} | {rule_e:.0%} | {zero_e:.0%} | {ft_s} | {sc:.0%} | {steps:.1f} |"
        )

    lines.extend(["", "## Per-fault (selected)", ""])
    for label, block in [
        ("Base N=6", base_eval),
        ("Zero-shot N=8", zero_by_n.get("8")),
        ("FT N=8", ft_by_n.get("8")),
        ("Zero-shot N=10", zero_by_n.get("10")),
        ("FT N=10", ft_by_n.get("10")),
    ]:
        if not block:
            continue
        lines.append(f"### {label}")
        lines.append(f"- Exact {_exact(block):.0%}, shortcut {block['shortcut']['shortcut_rate']:.0%}, "
                      f"timeout {block['timeout_rate']:.0%}")
        lines.append(f"- Device {_pf(block, 'device_failure')}")
        lines.append(f"- Perf {_pf(block, 'performance_degradation')}")
        lines.append(f"- Misconfig {_pf(block, 'misconfiguration')}")
        lines.append(f"- Link {_pf(block, 'link_failure')}")
        lines.append("")

    if "retain6_after_8" in ft_by_n or "retain6_after_10" in ft_by_n:
        lines.append("## N=6 retention after FT")
        for k in ("retain6_after_8", "retain6_after_10"):
            if k in ft_by_n:
                lines.append(f"- {k}: Exact {_exact(ft_by_n[k]):.0%}")
        lines.append("")

    gate8 = _go({}, ft_by_n["8"]) if "8" in ft_by_n else {"decision": "n/a"}
    lines.append(f"- FT N=8 hard-fault gate decision: **{gate8.get('decision')}**")
    lines.append("")
    lines.append("Run: `python -m src.run_star_scale_poc`")
    lines.append("")

    reports = PROJECT_ROOT / "reports"
    results_dir = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (reports / "star_scale_poc.md").write_text("\n".join(lines), encoding="utf-8")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("records",)}
        if isinstance(o, list):
            return [_clean(x) for x in o]
        return o

    (results_dir / "star_scale_poc.json").write_text(
        json.dumps(_clean(results), indent=2, default=str),
        encoding="utf-8",
    )
    print("Wrote", reports / "star_scale_poc.md")


if __name__ == "__main__":
    main()
