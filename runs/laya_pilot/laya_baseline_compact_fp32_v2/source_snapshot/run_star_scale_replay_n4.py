"""Extra: replay FT targeting N=4 (mix N=6 replay + N=4 demos)."""
from __future__ import annotations

import json
from pathlib import Path

from sb3_contrib import MaskablePPO

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
    p = load_protocol()
    p.setdefault("star_scale_poc", {"topology_types": ["star"]})
    pad = 12
    set_seeds(1)

    env = make_masked_base_env(["star"], protocol=p, **_env_kwargs(p, 6, pad))
    model = MaskablePPO.load("models/checkpoints/masked_ppo_bc_rule_pad12", env=env)
    env.close()

    print("collect replay6 + n4")
    r = collect_rule_demos(p, n_devices=6, pad=pad, n_ok=500, seed0=35_000)
    t = collect_rule_demos(p, n_devices=4, pad=pad, n_ok=300, seed0=44_000)
    print("r", r[3])
    print("t", t[3])
    obs = r[0] + t[0]
    acts = r[1] + t[1]
    masks = r[2] + t[2]
    print("mix", len(acts))
    losses = bc_train(model, obs, acts, masks, epochs=15, batch=64, lr=1e-4)
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "masked_ppo_bc_rule_pad12_replay_ft4.zip"
    model.save(str(ckpt.with_suffix("")))

    out = {"ft4_losses": losses}
    for n in [4, 6, 8, 10]:
        key = f"ft4_n{n}"
        out[key] = evaluate_model(
            model, p, n_devices=n, pad=pad, n=100, seed_base=109000 + 5000 + n
        )
        print(
            f"FT4->N{n} exact={_exact(out[key]):.0%} "
            f"Device={_pf(out[key], 'device_failure')} "
            f"Perf={_pf(out[key], 'performance_degradation')}"
        )

    env = make_masked_base_env(["star"], protocol=p, **_env_kwargs(p, 6, pad))
    base = MaskablePPO.load("models/checkpoints/masked_ppo_bc_rule_pad12", env=env)
    env.close()
    out["zero_n4"] = evaluate_model(
        base, p, n_devices=4, pad=pad, n=100, seed_base=109000 + 8004
    )
    out["rule_n4"] = evaluate_rule(p, n_devices=4, pad=pad, n=100, seed_base=109000 + 1004)
    print("zero4", _exact(out["zero_n4"]), "rule4", _exact(out["rule_n4"]))

    slim = {}
    for k, v in out.items():
        if k == "ft4_losses":
            slim[k] = v
            continue
        slim[k] = {
            "exact": _exact(v),
            "per_fault": v.get("per_fault"),
            "shortcut": v.get("shortcut"),
            "timeout_rate": v.get("timeout_rate"),
            "avg_steps": v.get("stats", {}).get("avg_steps_all"),
        }
    Path("results/star_scale_replay_n4_extra.json").write_text(
        json.dumps(slim, indent=2, default=str), encoding="utf-8"
    )

    # patch main replay report
    main_report = PROJECT_ROOT / "reports" / "star_scale_replay_poc.md"
    extra = [
        "",
        "## N=4 targeted Replay-FT (extra)",
        "",
        f"- Mix: 500×N6 + 300×N4, 15 epochs, lr=1e-4",
        f"- Rule N=4: {_exact(out['rule_n4']):.0%}",
        f"- Zero-shot N=4: {_exact(out['zero_n4']):.0%} (N=6-trained policy does **not** transfer down)",
        f"- Replay-FT4 → N=4: **{_exact(out['ft4_n4']):.0%}**",
        f"- Retain N=6: {_exact(out['ft4_n6']):.0%}",
        f"- Side N=8/10: {_exact(out['ft4_n8']):.0%} / {_exact(out['ft4_n10']):.0%}",
        "",
        "N=4 zero-shot failure mode: after scan, policy loops `check_status(center)` / "
        "`ping(center→edge_2)` then wrong misconfig/perf — N=6 spoke habits, not the 3-spoke tree.",
        "",
    ]
    text = main_report.read_text(encoding="utf-8")
    if "## N=4 targeted Replay-FT" not in text:
        main_report.write_text(text.rstrip() + "\n" + "\n".join(extra), encoding="utf-8")
    print("Wrote extra + updated report")


if __name__ == "__main__":
    main()
