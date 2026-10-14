"""E3 — Stage A collapse diagnosis + Stage B ent_coef=0.01."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from netheal.environment.actions import ActionCategory
from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .action_mask import action_masks_for_env, base_env
from .action_trace import aggregate_behavior, run_episode_with_policy, write_trace_csv
from .diagnosis_decomposition import aggregate_decomposition
from .env_utils import make_masked_base_env, make_masked_eval_env, metrics_env
from .metrics_utils import summarize_evaluator
from .protocol import PROJECT_ROOT, load_protocol
from .run_diagnosis_gate_poc import shortcut_rate
from .run_symmetric_reward_poc import set_seeds

NONLINK = ("device_failure", "performance_degradation", "misconfiguration")
PROBE_STEPS = (2048, 5120, 10240, 20480, 30720)


def _selector(model: MaskablePPO, *, deterministic: bool):
    def choose(env, obs, _step, _records):
        masks = action_masks_for_env(env)
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
        return int(action)

    return choose


def rollout_stats(
    model: MaskablePPO,
    protocol: dict,
    *,
    n_episodes: int,
    seed_base: int,
    deterministic: bool,
    agent: str,
) -> Dict:
    cfg = protocol["e3_exploration_poc"]
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    evaluator = CompetitionEvaluator()
    records: List = []
    summaries: List = []
    post_scan_tool = 0
    post_scan_dx = 0
    post_scan_n = 0
    n_scan = 0
    n_actions = 0

    for i in range(n_episodes):
        fault = faults[i % len(faults)]
        env = make_masked_eval_env(
            cfg["topology_types"], fault_types=[fault], protocol=protocol
        )
        recs, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed_base + i,
            agent=agent,
            choose_action=_selector(model, deterministic=deterministic),
        )
        m = metrics_env(env).last_episode_metrics
        if m:
            evaluator.add_episode_metrics(m)
        saw_scan = False
        for rec in recs:
            n_actions += 1
            if rec.action_type == "scan_network":
                n_scan += 1
                saw_scan = True
                continue
            if not saw_scan:
                continue
            post_scan_n += 1
            if rec.action_category == ActionCategory.DIAGNOSTIC.value:
                post_scan_tool += 1
            elif rec.action_category == ActionCategory.DIAGNOSIS.value:
                post_scan_dx += 1
        records.extend(recs)
        summaries.append(summary)
        env.close()

    behavior = aggregate_behavior(summaries, records)
    decomp = aggregate_decomposition(summaries)
    nl = [s for s in summaries if s.fault_type in NONLINK]
    tool_nl = (
        float(np.mean([s.tool_actions_before_diagnosis >= 2 for s in nl])) if nl else 0.0
    )
    # scan counts as a tool; "diagnostic before dx" = tools besides the opening scan
    diag_before = (
        float(
            np.mean(
                [
                    any(
                        r.episode_id == s.episode_id and r.action_category == "diagnostic"
                        for r in records
                    )
                    for s in nl
                ]
            )
        )
        if nl
        else 0.0
    )
    return {
        "deterministic": deterministic,
        "stats": summarize_evaluator(evaluator),
        "behavior": behavior,
        "decomposition": {
            "fault_type_accuracy": decomp["fault_type_accuracy"],
            "location_accuracy": decomp["location_accuracy"],
            "exact_diagnosis_accuracy": decomp["exact_diagnosis_accuracy"],
        },
        "per_fault_dsr": behavior.get("fault_type_dsr", {}),
        "shortcut": shortcut_rate(summaries, records),
        "avg_steps": float(np.mean([s.steps for s in summaries])) if summaries else 0.0,
        "tool_use_before_dx_nonlink": diag_before,
        "p_scan": n_scan / max(1, n_actions),
        "p_tool_given_post_scan": post_scan_tool / max(1, post_scan_n),
        "p_diagnosis_given_post_scan": post_scan_dx / max(1, post_scan_n),
        "records": records,
        "_tool_nl_ge2": tool_nl,
    }


def _logger_scalars(model) -> tuple:
    kv = getattr(getattr(model, "logger", None), "name_to_value", {}) or {}
    ent = kv.get("train/entropy_loss")
    eplen = kv.get("rollout/ep_len_mean")
    return (None if ent is None else float(ent), None if eplen is None else float(eplen))


def probe_checkpoint(model, protocol: dict, timesteps: int, history: List[dict], n_probe: int = 16) -> None:
    ent, eplen = _logger_scalars(model)
    stats = rollout_stats(
        model,
        protocol,
        n_episodes=n_probe,
        seed_base=88000 + int(timesteps),
        deterministic=False,
        agent=f"probe_{timesteps}",
    )
    stats.pop("records", None)
    stats.pop("behavior", None)
    row = {
        "timesteps": int(timesteps),
        "entropy_loss": ent,
        "ep_len_mean_sb3": eplen,
        "avg_steps_stochastic": stats["avg_steps"],
        "p_scan": stats["p_scan"],
        "p_tool_given_post_scan": stats["p_tool_given_post_scan"],
        "p_diagnosis_given_post_scan": stats["p_diagnosis_given_post_scan"],
        "tool_use_before_dx_nonlink": stats["tool_use_before_dx_nonlink"],
    }
    history.append(row)
    print(f"[probe] t={timesteps} {row}")


class StageACallback(BaseCallback):
    def __init__(self, protocol: dict, history: List[dict], n_probe: int = 16):
        super().__init__()
        self.protocol = protocol
        self.history = history
        self.n_probe = n_probe
        self._fired = set()

    def _on_step(self) -> bool:
        ts = int(self.num_timesteps)
        target = None
        for p in PROBE_STEPS:
            if ts >= p and p not in self._fired:
                target = p
                break
        if target is None:
            return True
        self._fired.add(target)
        probe_checkpoint(self.model, self.protocol, target, self.history, self.n_probe)
        return True


def train_model(protocol: dict, *, ent_coef: float, out_path: Path, history: List[dict]) -> MaskablePPO:
    cfg = protocol["e3_exploration_poc"]
    set_seeds(cfg["policy_seed"])
    env = make_masked_base_env(cfg["topology_types"], protocol=protocol)
    check_env(env, warn=True)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=cfg["policy_seed"],
        ent_coef=ent_coef,
    )
    print(f"ent_coef={model.ent_coef}")
    cb = StageACallback(protocol, history)
    model.learn(total_timesteps=cfg["timesteps"], callback=cb)
    if not history or history[-1]["timesteps"] < min(cfg["timesteps"], model.num_timesteps):
        probe_checkpoint(model, protocol, int(model.num_timesteps), history)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path.with_suffix("")))
    env.close()
    return model


def go_nogo(control_stoch: Dict, e3_stoch: Dict) -> Dict:
    steps = e3_stoch["avg_steps"]
    tools = e3_stoch["tool_use_before_dx_nonlink"]
    nogo = steps < 3.0 and tools < 0.10
    return {
        "e3_avg_steps": steps,
        "e3_tool_use_nonlink": tools,
        "control_avg_steps": control_stoch["avg_steps"],
        "control_tool_use_nonlink": control_stoch["tool_use_before_dx_nonlink"],
        "nogo": nogo,
        "decision": "no_go" if nogo else "go",
        "rule": "No-Go if AvgSteps < 3.0 AND non-link tool-use-before-dx < 10% (stochastic)",
    }


def build_report(
    c_det, e_det, c_stoch, e_stoch, hist_c, hist_e, gate, cfg
) -> str:
    lines = [
        "# E3 — Exploration Collapse + ent_coef=0.01 (seed 1)",
        "",
        "> Reward/action unchanged. E1/E2 not mixed. Success = tool-use trajectories, not DSR.",
        "> Go/No-Go uses **stochastic** eval. Deterministic DSR is reference only.",
        "",
        f"Train: Star, 6 devices, MaskablePPO, {cfg['timesteps']}, seed {cfg['policy_seed']}",
        "Control `ent_coef=0.0` (SB3 default). E3 `ent_coef=0.01`.",
        "",
        "## Stage A — Control action mix (stochastic probes)",
        "",
        "| t | entropy_loss | sb3 ep_len | avg_steps | P(tool\\|post-scan) | P(dx\\|post-scan) | tool-use NL |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in hist_c:
        lines.append(
            f"| {r['timesteps']} | {r['entropy_loss']} | {r['ep_len_mean_sb3']} | "
            f"{r['avg_steps_stochastic']:.2f} | {r['p_tool_given_post_scan']:.2f} | "
            f"{r['p_diagnosis_given_post_scan']:.2f} | {r['tool_use_before_dx_nonlink']:.0%} |"
        )
    lines += [
        "",
        "## Stage B — E3 probes",
        "",
        "| t | entropy_loss | sb3 ep_len | avg_steps | P(tool\\|post-scan) | P(dx\\|post-scan) | tool-use NL |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in hist_e:
        lines.append(
            f"| {r['timesteps']} | {r['entropy_loss']} | {r['ep_len_mean_sb3']} | "
            f"{r['avg_steps_stochastic']:.2f} | {r['p_tool_given_post_scan']:.2f} | "
            f"{r['p_diagnosis_given_post_scan']:.2f} | {r['tool_use_before_dx_nonlink']:.0%} |"
        )
    lines += [
        "",
        "## Final comparison",
        "",
        "| Metric | Control sto | E3 sto | Control det | E3 det |",
        "|---|---:|---:|---:|---:|",
        f"| AvgSteps | {c_stoch['avg_steps']:.2f} | {e_stoch['avg_steps']:.2f} | "
        f"{c_det['avg_steps']:.2f} | {e_det['avg_steps']:.2f} |",
        f"| Tool-use before dx (non-link) | {c_stoch['tool_use_before_dx_nonlink']:.0%} | "
        f"{e_stoch['tool_use_before_dx_nonlink']:.0%} | "
        f"{c_det['tool_use_before_dx_nonlink']:.0%} | {e_det['tool_use_before_dx_nonlink']:.0%} |",
        f"| P(tool\\|post-scan) | {c_stoch['p_tool_given_post_scan']:.2f} | "
        f"{e_stoch['p_tool_given_post_scan']:.2f} | "
        f"{c_det['p_tool_given_post_scan']:.2f} | {e_det['p_tool_given_post_scan']:.2f} |",
        f"| P(dx\\|post-scan) | {c_stoch['p_diagnosis_given_post_scan']:.2f} | "
        f"{e_stoch['p_diagnosis_given_post_scan']:.2f} | "
        f"{c_det['p_diagnosis_given_post_scan']:.2f} | {e_det['p_diagnosis_given_post_scan']:.2f} |",
        f"| Exact DSR | {c_stoch['decomposition']['exact_diagnosis_accuracy']:.0%} | "
        f"{e_stoch['decomposition']['exact_diagnosis_accuracy']:.0%} | "
        f"{c_det['decomposition']['exact_diagnosis_accuracy']:.0%} | "
        f"{e_det['decomposition']['exact_diagnosis_accuracy']:.0%} |",
        f"| Shortcut (det-style) | {c_stoch['shortcut']['shortcut_rate']:.0%} | "
        f"{e_stoch['shortcut']['shortcut_rate']:.0%} | "
        f"{c_det['shortcut']['shortcut_rate']:.0%} | {e_det['shortcut']['shortcut_rate']:.0%} |",
        "",
        "## Go/No-Go (stochastic)",
        "",
        f"- E3 AvgSteps: **{gate['e3_avg_steps']:.2f}** (need ≥ 3.0)",
        f"- E3 non-link tool-use: **{gate['e3_tool_use_nonlink']:.0%}** (need ≥ 10%)",
        f"- Decision: **{gate['decision']}**",
        f"- {gate['rule']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()

    protocol = load_protocol()
    cfg = protocol["e3_exploration_poc"]
    n_eval = args.eval_episodes or cfg.get("eval_episodes", 100)
    seed_det = cfg.get("eval_seed_base", 100000)
    seed_stoch = seed_det + 5000
    ckpt = PROJECT_ROOT / "models" / "checkpoints"
    control_path = ckpt / "masked_ppo_e3_control_star.zip"
    e3_path = ckpt / "masked_ppo_e3_ent001_star.zip"
    hist_c: List[dict] = []
    hist_e: List[dict] = []

    if args.skip_train and control_path.with_suffix(".zip").exists():
        print("Loading Stage A control...")
        control = MaskablePPO.load(str(control_path.with_suffix("")))
    else:
        print("Stage A: training control ent_coef=0.0 + probes...")
        control = train_model(protocol, ent_coef=0.0, out_path=control_path, history=hist_c)
        (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "results" / "e3_stage_a_probes.json").write_text(
            json.dumps(hist_c, indent=2), encoding="utf-8"
        )

    if args.skip_train and e3_path.with_suffix(".zip").exists():
        print("Loading E3...")
        e3 = MaskablePPO.load(str(e3_path.with_suffix("")))
    else:
        print("Stage B: training E3 ent_coef=0.01 + probes...")
        e3 = train_model(protocol, ent_coef=0.01, out_path=e3_path, history=hist_e)

    print("Stochastic eval...")
    c_stoch = rollout_stats(
        control, protocol, n_episodes=n_eval, seed_base=seed_stoch, deterministic=False, agent="control_stoch"
    )
    e_stoch = rollout_stats(
        e3, protocol, n_episodes=n_eval, seed_base=seed_stoch, deterministic=False, agent="e3_stoch"
    )
    print("Deterministic eval (DSR reference)...")
    c_det = rollout_stats(
        control, protocol, n_episodes=n_eval, seed_base=seed_det, deterministic=True, agent="control_det"
    )
    e_det = rollout_stats(
        e3, protocol, n_episodes=n_eval, seed_base=seed_det, deterministic=True, agent="e3_det"
    )

    results = PROJECT_ROOT / "results"
    write_trace_csv(results / "e3_control_stoch_trace.csv", c_stoch.pop("records"))
    write_trace_csv(results / "e3_variant_stoch_trace.csv", e_stoch.pop("records"))
    write_trace_csv(results / "e3_control_det_trace.csv", c_det.pop("records"))
    write_trace_csv(results / "e3_variant_det_trace.csv", e_det.pop("records"))
    c_stoch.pop("behavior", None)
    e_stoch.pop("behavior", None)
    c_det.pop("behavior", None)
    e_det.pop("behavior", None)

    gate = go_nogo(c_stoch, e_stoch)
    report = build_report(c_det, e_det, c_stoch, e_stoch, hist_c, hist_e, gate, cfg)
    (PROJECT_ROOT / "reports" / "e3_exploration_poc.md").write_text(report, encoding="utf-8")
    (results / "e3_exploration_poc.json").write_text(
        json.dumps(
            {
                "stage_a_control": hist_c,
                "stage_b_e3": hist_e,
                "control_stoch": c_stoch,
                "e3_stoch": e_stoch,
                "control_det": c_det,
                "e3_det": e_det,
                "go_nogo": gate,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"E3 sto steps={e_stoch['avg_steps']:.2f} tools={e_stoch['tool_use_before_dx_nonlink']:.0%} {gate['decision']}")
    print(f"Wrote {PROJECT_ROOT / 'reports' / 'e3_exploration_poc.md'}")


if __name__ == "__main__":
    main()
