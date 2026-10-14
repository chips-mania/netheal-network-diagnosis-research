"""Star scale replay FT: mix N=6 demos when fine-tuning; also eval N=4.

Loads/trains pad12 base on N=6, then FT with (replay6 ∪ target_N) demos.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_checker import check_env

from .env_utils import make_masked_base_env
from .protocol import PROJECT_ROOT, load_protocol
from .run_star_scale_poc import (
    _env_kwargs,
    _exact,
    _pf,
    bc_train,
    collect_rule_demos,
    evaluate_model,
    evaluate_rule,
)
from .run_symmetric_reward_poc import set_seeds


def _concat_demos(*bundles: Tuple[list, list, list, dict]):
    obs, acts, masks = [], [], []
    stats = []
    for o, a, m, s in bundles:
        obs.extend(o)
        acts.extend(a)
        masks.extend(m)
        stats.append(s)
    return obs, acts, masks, stats


def main() -> None:
    protocol = load_protocol()
    cfg = protocol.setdefault(
        "star_scale_replay_poc",
        {
            "topology_types": ["star"],
            "pad_max_devices": 12,
            "n_list": [4, 6, 8, 10],
            "base_n": 6,
            "base_demos": 2000,
            "base_bc_epochs": 40,
            "replay6_demos": 500,
            "ft_target_demos": 300,
            "ft_bc_epochs": 15,
            "ft_lr": 0.0001,
            "batch_size": 64,
            "policy_seed": 1,
            "eval_episodes": 100,
            "eval_seed_base": 109000,
        },
    )
    # keep star_scale_poc topology key used by helpers
    protocol.setdefault("star_scale_poc", {"topology_types": ["star"]})
    protocol["star_scale_poc"]["topology_types"] = cfg["topology_types"]

    pad = int(cfg["pad_max_devices"])
    base_n = int(cfg["base_n"])
    n_eval = int(cfg["eval_episodes"])
    seed = int(cfg["policy_seed"])
    set_seeds(seed)
    topologies = cfg["topology_types"]
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_path = ckpt_dir / "masked_ppo_bc_rule_pad12.zip"

    results: Dict = {"pad": pad, "base_n": base_n, "mode": "replay_mix"}

    # --- Base ---
    print(f"=== Base BC N={base_n} pad={pad} ===")
    obs6, acts6, masks6, collect6 = collect_rule_demos(
        protocol, n_devices=base_n, pad=pad, n_ok=int(cfg["base_demos"]), seed0=30_000
    )
    print("base collect", collect6)
    env = make_masked_base_env(topologies, protocol=protocol, **_env_kwargs(protocol, base_n, pad))
    check_env(env, warn=True)
    model = MaskablePPO("MultiInputPolicy", env, verbose=0, seed=seed)
    losses = bc_train(
        model,
        obs6,
        acts6,
        masks6,
        epochs=int(cfg["base_bc_epochs"]),
        batch=int(cfg["batch_size"]),
        lr=3e-4,
    )
    model.save(str(base_path.with_suffix("")))
    env.close()
    results["base_collect"] = collect6
    results["base_bc_losses"] = losses

    # Keep a subset of N=6 transitions for replay (first K episodes worth ≈ by transitions)
    # Re-collect dedicated replay set for clarity
    print(f"=== Replay pool N=6 demos={cfg['replay6_demos']} ===")
    r_obs, r_acts, r_masks, r_stats = collect_rule_demos(
        protocol,
        n_devices=base_n,
        pad=pad,
        n_ok=int(cfg["replay6_demos"]),
        seed0=35_000,
    )
    print("replay6", r_stats)
    results["replay6_collect"] = r_stats

    n_list = [int(x) for x in cfg["n_list"]]

    # --- Rule + zero-shot ---
    rule_by_n: Dict[str, Dict] = {}
    zero_by_n: Dict[str, Dict] = {}
    for n_dev in n_list:
        print(f"=== Rule N={n_dev} ===")
        rule_by_n[str(n_dev)] = evaluate_rule(
            protocol,
            n_devices=n_dev,
            pad=pad,
            n=n_eval,
            seed_base=int(cfg["eval_seed_base"]) + 1000 * n_dev,
        )
        print(f"  rule exact={_exact(rule_by_n[str(n_dev)]):.0%}")
        print(f"=== Zero-shot N={n_dev} ===")
        zero_by_n[str(n_dev)] = evaluate_model(
            model,
            protocol,
            n_devices=n_dev,
            pad=pad,
            n=n_eval,
            seed_base=int(cfg["eval_seed_base"]) + 2000 * n_dev,
        )
        print(f"  zero exact={_exact(zero_by_n[str(n_dev)]):.0%}")
    results["rule"] = rule_by_n
    results["zeroshot"] = zero_by_n

    # --- Replay FT: from base, FT to 8 then 10 with N=6 mix ---
    # Reload fresh base before FT chain so comparison is clean
    env = make_masked_base_env(topologies, protocol=protocol, **_env_kwargs(protocol, base_n, pad))
    model = MaskablePPO.load(str(base_path), env=env)
    env.close()

    ft_by_n: Dict[str, Dict] = {}
    ft_meta: Dict[str, Dict] = {}

    for n_dev in [8, 10]:
        print(
            f"=== Replay-FT N={n_dev}: "
            f"{cfg['replay6_demos']}×N6 + {cfg['ft_target_demos']}×N{n_dev} ==="
        )
        t_obs, t_acts, t_masks, t_stats = collect_rule_demos(
            protocol,
            n_devices=n_dev,
            pad=pad,
            n_ok=int(cfg["ft_target_demos"]),
            seed0=40_000 + 1000 * n_dev,
        )
        print("  target collect", t_stats)
        obs_m, acts_m, masks_m, mix_stats = _concat_demos(
            (r_obs, r_acts, r_masks, r_stats),
            (t_obs, t_acts, t_masks, t_stats),
        )
        print(f"  mix transitions={len(acts_m)}")
        losses_f = bc_train(
            model,
            obs_m,
            acts_m,
            masks_m,
            epochs=int(cfg["ft_bc_epochs"]),
            batch=int(cfg["batch_size"]),
            lr=float(cfg["ft_lr"]),
        )
        ft_path = ckpt_dir / f"masked_ppo_bc_rule_pad12_replay_ft{n_dev}.zip"
        model.save(str(ft_path.with_suffix("")))
        ft_meta[str(n_dev)] = {
            "target_collect": t_stats,
            "mix_transitions": len(acts_m),
            "losses": losses_f,
            "ckpt": str(ft_path),
        }

        # eval all sizes after this FT step
        for eval_n in n_list:
            key = f"after_ft{n_dev}_n{eval_n}"
            print(f"  Eval after FT{n_dev} on N={eval_n}...")
            ft_by_n[key] = evaluate_model(
                model,
                protocol,
                n_devices=eval_n,
                pad=pad,
                n=n_eval,
                seed_base=int(cfg["eval_seed_base"]) + 3000 * n_dev + eval_n,
            )
            print(f"    exact={_exact(ft_by_n[key]):.0%}")

    results["ft_meta"] = ft_meta
    results["ft_evals"] = ft_by_n

    # Snapshot "best" FT8 and FT10 rows for table
    table_ft = {
        "4": ft_by_n.get("after_ft8_n4") or ft_by_n.get("after_ft10_n4"),
        "6": ft_by_n.get("after_ft8_n6"),
        "8": ft_by_n.get("after_ft8_n8"),
        "10": ft_by_n.get("after_ft10_n10"),
    }

    lines = [
        "# Star scale replay FT (+ N=4)",
        "",
        "> Pad=12. Base Rule-BC on N=6. Fine-tune mixes **N=6 replay demos** with target-N demos.",
        "> Contrast: previous few-shot FT without replay forgot N=6 and hurt Exact.",
        "",
        f"- Base demos: {collect6['ok']}, last loss {losses[-1]:.4f}",
        f"- Replay N=6: {r_stats['ok']} ep; target: {cfg['ft_target_demos']} ep; "
        f"FT epochs {cfg['ft_bc_epochs']}, lr={cfg['ft_lr']}",
        "",
        "## Summary (Exact DSR)",
        "",
        "| N | Rule | Zero-shot | Replay-FT (after FT8 / FT10*) |",
        "|---:|---:|---:|---:|",
    ]
    for n_dev in n_list:
        ns = str(n_dev)
        rule_e = _exact(rule_by_n[ns])
        zero_e = _exact(zero_by_n[ns])
        if n_dev <= 8:
            ft_v = ft_by_n.get(f"after_ft8_n{n_dev}")
        else:
            ft_v = ft_by_n.get(f"after_ft10_n{n_dev}")
        ft_s = f"{_exact(ft_v):.0%}" if ft_v else "—"
        # also show FT10 column note for n<=8 retention
        if n_dev <= 8 and f"after_ft10_n{n_dev}" in ft_by_n:
            ft_s += f" / {_exact(ft_by_n[f'after_ft10_n{n_dev}']):.0%}†"
        lines.append(f"| {n_dev} | {rule_e:.0%} | {zero_e:.0%} | {ft_s} |")

    lines.extend(
        [
            "",
            "\\* Table FT cell: for N≤8 = after FT8 (and † after FT10 if present); for N=10 = after FT10.",
            "",
            "## vs naive FT (no replay, prior run)",
            "",
            "| N | Zero-shot | Naive FT | Replay-FT |",
            "|---:|---:|---:|---:|",
            "| 8 | 76% | 66% | "
            + (f"{_exact(ft_by_n['after_ft8_n8']):.0%}" if "after_ft8_n8" in ft_by_n else "—")
            + " |",
            "| 10 | 64% | 45% | "
            + (f"{_exact(ft_by_n['after_ft10_n10']):.0%}" if "after_ft10_n10" in ft_by_n else "—")
            + " |",
            "| 6 retain | 97% | 50%→35% | "
            + (
                f"{_exact(ft_by_n['after_ft8_n6']):.0%}→{_exact(ft_by_n['after_ft10_n6']):.0%}"
                if "after_ft8_n6" in ft_by_n and "after_ft10_n6" in ft_by_n
                else "—"
            )
            + " |",
            "",
            "## Per-fault highlights",
            "",
        ]
    )

    for label, key in [
        ("Zero-shot N=4", "4"),
        ("Replay-FT8 on N=4", "after_ft8_n4"),
        ("Replay-FT8 on N=8", "after_ft8_n8"),
        ("Replay-FT10 on N=10", "after_ft10_n10"),
        ("Retain N=6 after FT10", "after_ft10_n6"),
    ]:
        block = zero_by_n.get(key) if key in zero_by_n else ft_by_n.get(key)
        if not block:
            continue
        lines.append(f"### {label}")
        lines.append(
            f"- Exact {_exact(block):.0%}, shortcut {block['shortcut']['shortcut_rate']:.0%}"
        )
        lines.append(f"- Device {_pf(block, 'device_failure')}")
        lines.append(f"- Perf {_pf(block, 'performance_degradation')}")
        lines.append(f"- Misconfig {_pf(block, 'misconfiguration')}")
        lines.append(f"- Link {_pf(block, 'link_failure')}")
        lines.append("")

    lines.append("Run: `python -m src.run_star_scale_replay`")
    lines.append("")

    reports = PROJECT_ROOT / "reports"
    results_dir = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (reports / "star_scale_replay_poc.md").write_text("\n".join(lines), encoding="utf-8")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("records",)}
        if isinstance(o, list):
            return [_clean(x) for x in o]
        return o

    (results_dir / "star_scale_replay_poc.json").write_text(
        json.dumps(_clean(results), indent=2, default=str),
        encoding="utf-8",
    )
    print("Wrote", reports / "star_scale_replay_poc.md")


if __name__ == "__main__":
    main()
