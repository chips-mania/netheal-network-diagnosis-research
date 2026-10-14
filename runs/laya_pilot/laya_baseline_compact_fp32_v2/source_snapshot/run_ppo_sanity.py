"""Step 9: Star-only PPO sanity training and Go/No-Go vs Random-Valid."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .agents import random_valid_action
from .env_utils import make_base_env, make_eval_env
from .metrics_utils import format_summary_row, summarize_evaluator
from .protocol import PROJECT_ROOT, load_protocol


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_ppo(protocol: dict, out_dir: Path) -> PPO:
    cfg = protocol["ppo_sanity"]
    set_seeds(cfg["policy_seed"])
    env = make_base_env(cfg["topology_types"], protocol=protocol)
    check_env(env, warn=True)
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=cfg["policy_seed"],
    )
    model.learn(total_timesteps=cfg["timesteps"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "ppo_sanity_star.zip"
    model.save(str(model_path.with_suffix("")))
    env.close()
    return model


def eval_ppo(model: PPO, protocol: dict, n_episodes: int = 40) -> dict:
    evaluator = CompetitionEvaluator()
    cfg = protocol["ppo_sanity"]
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    for i in range(n_episodes):
        fault = faults[i % len(faults)]
        env = make_eval_env(cfg["topology_types"], fault_types=[fault], protocol=protocol)
        obs, _ = env.reset(seed=50000 + i)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
        if env.last_episode_metrics is not None:
            evaluator.add_episode_metrics(env.last_episode_metrics)
        env.close()
    return summarize_evaluator(evaluator)


def eval_random_valid_star(protocol: dict, n_episodes: int = 40) -> dict:
    evaluator = CompetitionEvaluator()
    cfg = protocol["ppo_sanity"]
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    rng = random.Random(0)
    for i in range(n_episodes):
        fault = faults[i % len(faults)]
        env = make_eval_env(cfg["topology_types"], fault_types=[fault], protocol=protocol)
        obs, _ = env.reset(seed=50000 + i)
        max_steps = env.unwrapped.max_episode_steps
        for step in range(max_steps):
            prefer = step >= max_steps - 1
            action = random_valid_action(env, rng, prefer_diagnosis=prefer)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        if env.last_episode_metrics is not None:
            evaluator.add_episode_metrics(env.last_episode_metrics)
        env.close()
    return summarize_evaluator(evaluator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-episodes", type=int, default=40)
    args = parser.parse_args()

    protocol = load_protocol()
    model_dir = PROJECT_ROOT / "models" / "checkpoints"
    report_path = PROJECT_ROOT / "reports" / "ppo_sanity_gonogo.md"

    print("Training PPO (Star-only sanity)...")
    model = train_ppo(protocol, model_dir)

    print("Evaluating PPO vs Random-Valid on Star...")
    ppo_stats = eval_ppo(model, protocol, n_episodes=args.eval_episodes)
    rv_stats = eval_random_valid_star(protocol, n_episodes=args.eval_episodes)

    dsr_gap = ppo_stats["dsr"] - rv_stats["dsr"]
    go = ppo_stats["dsr"] > rv_stats["dsr"] and ppo_stats["dsr"] > 0.05

    lines = [
        "# PPO Sanity Go/No-Go",
        "",
        format_summary_row("PPO (Star eval)", ppo_stats),
        format_summary_row("Random-Valid (Star eval)", rv_stats),
        "",
        f"- DSR gap (PPO - Random-Valid): {dsr_gap:.3f}",
        f"- **Decision: {'GO' if go else 'NO-GO'}**",
        "",
        "GO criteria: PPO DSR > Random-Valid DSR and PPO DSR > 0.05",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for line in lines:
        print(line)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
