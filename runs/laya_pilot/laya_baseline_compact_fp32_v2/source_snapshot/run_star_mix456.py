"""Train Rule-BC on mixed Star N={4,5,6}, zero-shot eval on N=8 (and neighbors).

Compares to N=6-only pad12 base zero-shot (prior star_scale runs).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

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


def main() -> None:
    protocol = load_protocol()
    protocol.setdefault("star_scale_poc", {"topology_types": ["star"]})
    cfg = protocol.setdefault(
        "star_mix456_poc",
        {
            "topology_types": ["star"],
            "pad_max_devices": 12,
            "train_ns": [4, 5, 6],
            "demos_per_n": 700,
            "bc_epochs": 40,
            "batch_size": 64,
            "policy_seed": 1,
            "eval_episodes": 100,
            "eval_ns": [4, 5, 6, 8, 10],
            "eval_seed_base": 110000,
        },
    )
    protocol["star_scale_poc"]["topology_types"] = cfg["topology_types"]

    pad = int(cfg["pad_max_devices"])
    set_seeds(int(cfg["policy_seed"]))
    topologies = cfg["topology_types"]
    demos_per = int(cfg["demos_per_n"])

    obs_all: List = []
    acts_all: List = []
    masks_all: List = []
    collects = {}
    for n in cfg["train_ns"]:
        n = int(n)
        print(f"=== Collect Rule demos N={n} x{demos_per} ===")
        o, a, m, st = collect_rule_demos(
            protocol,
            n_devices=n,
            pad=pad,
            n_ok=demos_per,
            seed0=50_000 + 1000 * n,
        )
        print(" ", st)
        collects[str(n)] = st
        obs_all.extend(o)
        acts_all.extend(a)
        masks_all.extend(m)

    print(f"total transitions={len(acts_all)}")
    # Build policy env at pad space (use N=6 for env factory; pad fixes dims)
    env = make_masked_base_env(
        topologies, protocol=protocol, **_env_kwargs(protocol, 6, pad)
    )
    check_env(env, warn=True)
    model = MaskablePPO("MultiInputPolicy", env, verbose=0, seed=int(cfg["policy_seed"]))
    print("BC train mix 4/5/6...")
    losses = bc_train(
        model,
        obs_all,
        acts_all,
        masks_all,
        epochs=int(cfg["bc_epochs"]),
        batch=int(cfg["batch_size"]),
        lr=3e-4,
    )
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_bc_rule_pad12_mix456.zip"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt.with_suffix("")))
    env.close()

    n_eval = int(cfg["eval_episodes"])
    seed_base = int(cfg["eval_seed_base"])
    rule: Dict[str, Dict] = {}
    zero: Dict[str, Dict] = {}
    for n in cfg["eval_ns"]:
        n = int(n)
        print(f"=== Rule N={n} ===")
        rule[str(n)] = evaluate_rule(
            protocol, n_devices=n, pad=pad, n=n_eval, seed_base=seed_base + 100 * n
        )
        print(f"  rule exact={_exact(rule[str(n)]):.0%}")
        print(f"=== Zero-shot mix456 on N={n} ===")
        zero[str(n)] = evaluate_model(
            model,
            protocol,
            n_devices=n,
            pad=pad,
            n=n_eval,
            seed_base=seed_base + 1000 * n,
        )
        print(f"  zero exact={_exact(zero[str(n)]):.0%}")

    # Prior baselines (from reports; fixed numbers for comparison table)
    prior_n6_only_zero = {
        "4": "0–22%",
        "5": "—",
        "6": "97–99%",
        "8": "76–79%",
        "10": "54–64%",
    }

    lines = [
        "# Mix N=4/5/6 Rule-BC → zero-shot N=8",
        "",
        "> Train: Rule demos on Star **N∈{4,5,6}** (pad=12), equal count per N.",
        "> Eval: greedy zero-shot on held-out sizes especially **N=8**.",
        "> Question: multi-size demos vs N=6-only path cloning.",
        "",
        f"- Demos/N: **{demos_per}** × {{4,5,6}} = {3 * demos_per} eps, "
        f"transitions={len(acts_all)}",
        f"- Per-N collect: {collects}",
        f"- Last BC loss: {losses[-1]:.4f}",
        f"- ckpt: `{ckpt.name}`",
        "",
        "## Exact DSR",
        "",
        "| N | Rule | Zero-shot mix4/5/6 | Zero-shot N=6-only (prior) | In train? |",
        "|---:|---:|---:|---:|---|",
    ]
    for n in cfg["eval_ns"]:
        ns = str(n)
        in_train = "yes" if int(n) in [int(x) for x in cfg["train_ns"]] else "**held-out**"
        prior = prior_n6_only_zero.get(ns, "—")
        lines.append(
            f"| {n} | {_exact(rule[ns]):.0%} | **{_exact(zero[ns]):.0%}** | {prior} | {in_train} |"
        )

    lines.extend(
        [
            "",
            "## Held-out N=8 detail (mix vs prior)",
            "",
            f"- mix4/5/6 zero-shot Exact: **{_exact(zero['8']):.0%}**",
            f"- N=6-only zero-shot Exact (prior): **76–79%**",
            f"- Rule ceiling N=8: {_exact(rule['8']):.0%}",
            f"- shortcut mix: {zero['8']['shortcut']['shortcut_rate']:.0%}",
            "",
            f"- Device {_pf(zero['8'], 'device_failure')}",
            f"- Perf {_pf(zero['8'], 'performance_degradation')}",
            f"- Misconfig {_pf(zero['8'], 'misconfiguration')}",
            f"- Link {_pf(zero['8'], 'link_failure')}",
            "",
            "## Train-set check N=4/5/6",
            "",
        ]
    )
    for n in [4, 5, 6]:
        ns = str(n)
        lines.append(
            f"- N={n}: Exact {_exact(zero[ns]):.0%}, "
            f"Device {_pf(zero[ns], 'device_failure')}, "
            f"Perf {_pf(zero[ns], 'performance_degradation')}"
        )

    delta = _exact(zero["8"]) - 0.775  # mid of 76-79
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- N=8 Exact − prior mid(77.5%): **{delta:+.0%}** absolute "
            f"({_exact(zero['8']):.0%} vs ~77.5%).",
            "- If mix ≫ N=6-only on N=8 **and** N=4 stays high: evidence for size-robust cloning of the Rule tree, not only N=6 path.",
            "- If mix ≈ N=6-only on N=8: multi-size demos alone do not buy held-out scale.",
            "",
            "Run: `python -m src.run_star_mix456`",
            "",
        ]
    )

    reports = PROJECT_ROOT / "reports"
    results_dir = PROJECT_ROOT / "results"
    reports.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (reports / "star_mix456_poc.md").write_text("\n".join(lines), encoding="utf-8")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k not in ("records",)}
        if isinstance(o, list):
            return [_clean(x) for x in o]
        return o

    payload = {
        "collects": collects,
        "n_transitions": len(acts_all),
        "bc_losses": losses,
        "rule": _clean(rule),
        "zeroshot": _clean(zero),
        "prior_n6_only_zero": prior_n6_only_zero,
    }
    (results_dir / "star_mix456_poc.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"N=8 mix exact={_exact(zero['8']):.0%} "
        f"N=4={_exact(zero['4']):.0%} N=6={_exact(zero['6']):.0%}"
    )
    print("Wrote", reports / "star_mix456_poc.md")


if __name__ == "__main__":
    main()
