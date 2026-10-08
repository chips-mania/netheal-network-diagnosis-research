"""Step 9.6-A/B/C: action mask audit and wrapper validation."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from netheal.faults.injector import FaultType

from netheal.environment.actions import ActionCategory

from .action_mask import base_env, masked_valid_action_ids, valid_action_mask, wrap_with_action_mask
from .agents import get_valid_action_ids, random_valid_action
from .env_utils import make_base_env, make_eval_env, make_masked_base_env
from .protocol import PROJECT_ROOT, load_protocol


AUDIT_MD = """# Action Mask Audit (Step 9.6-A)

## Question

Does `get_valid_actions()` use hidden fault ground truth, or only agent-observable state?

## Code path

1. `NetworkTroubleshootingEnv.get_valid_actions()` (`netheal/environment/env.py`)
2. → `StructuredActionSpace.get_valid_actions(discovered_devices)` (`netheal/environment/actions.py`)
3. → `_is_action_valid(spec, discovered_devices)`

`discovered_devices` comes from:

```python
self.observation.discovery_matrix.get_discovered_devices()
```

Devices enter the discovery matrix only via agent actions (scan, discover_neighbors, diagnostic side effects). Ground truth fault is **not** passed into validity logic.

## Ground truth usage elsewhere (not in mask)

| Component | Uses ground_truth? | Role |
|-----------|-------------------|------|
| `get_valid_actions()` | **No** | Action validity |
| `SparseRewardCalculator` | Yes | Reward only at diagnosis step |
| `info['ground_truth_fault']` | Yes | Eval/debug (agent should not receive in production MCP path) |
| Hint provider | Yes (optional) | Disabled in our protocol (`enable_user_hints=False`) |

## Validity rules (leakage-free)

| Category | Valid when |
|----------|------------|
| `scan_network` | Always |
| `discover_neighbors(device)` | `device` already discovered |
| `ping/traceroute` | source **and** destination discovered |
| `check_status/interfaces` | device discovered |
| `diagnosis(location)` | all devices in location discovered |

**Not used:** fault type, fault location, tool simulator hidden state.

## Does mask imply the correct fault?

No. After `scan_network`, many devices become discovered, so **many** diagnosis actions become valid simultaneously (all fault types × many locations). The mask does **not** narrow to the true fault.

Example at step 0 (nothing discovered): typically only `scan_network` is valid — no diagnosis.

## Workflow forcing?

The mask blocks semantically impossible actions (e.g. ping undiscovered device). It does **not** enforce scan → ping → diagnosis ordering. Multiple valid actions remain at each step.

## Conclusion

**Leakage-free for masking purposes.** Safe to use `get_valid_actions()` as the action mask source for MaskablePPO, with the caveat that this is an **environment–policy interface** fix (vanilla PPO ignored valid actions), not a NetHeal bug claim.

See also: `reports/mask_wrapper_validation.md`, `reports/masked_random_check.md`.
"""


def validate_mask_wrapper(protocol: dict, n_states: int = 500, seed: int = 0) -> dict:
    cfg = protocol["ppo_sanity"]
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    rng = random.Random(seed)

    shape_mismatch = 0
    empty_mask = 0
    mask_mismatch = 0
    samples_checked = 0

    for i in range(n_states):
        fault = faults[rng.randrange(len(faults))]
        env = wrap_with_action_mask(
            make_base_env(cfg["topology_types"], fault_types=[fault], protocol=protocol)
        )
        obs, _ = env.reset(seed=seed + i)
        del obs

        for _ in range(rng.randint(1, 4)):
            mask = valid_action_mask(env)
            n = env.action_space.n
            if mask.shape[0] != n:
                shape_mismatch += 1
            if not mask.any():
                empty_mask += 1

            direct = set(get_valid_action_ids(env))
            from_mask = set(masked_valid_action_ids(env))
            if direct != from_mask:
                mask_mismatch += 1
            samples_checked += 1

            valid_ids = list(from_mask)
            action = rng.choice(valid_ids)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        env.close()

    return {
        "n_episodes_sampled": n_states,
        "samples_checked": samples_checked,
        "shape_mismatch": shape_mismatch,
        "empty_mask": empty_mask,
        "mask_mismatch_vs_get_valid_actions": mask_mismatch,
        "pass": shape_mismatch == 0 and empty_mask == 0 and mask_mismatch == 0,
    }


def validate_masked_random(protocol: dict, n_episodes: int = 50, seed: int = 0) -> dict:
    """9.6-C: Random-Valid vs mask-indexed random should both use valid actions only."""
    cfg = protocol["ppo_sanity"]
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    rng_a = random.Random(seed)
    rng_b = random.Random(seed)

    invalid_a = 0
    invalid_b = 0
    steps_a = 0
    steps_b = 0

    for i in range(n_episodes):
        fault = faults[i % len(faults)]
        ep_seed = 60000 + i

        env_a = make_eval_env(cfg["topology_types"], fault_types=[fault], protocol=protocol)
        env_b = wrap_with_action_mask(
            make_eval_env(cfg["topology_types"], fault_types=[fault], protocol=protocol)
        )

        env_a.reset(seed=ep_seed)
        env_b.reset(seed=ep_seed)
        max_steps = base_env(env_a).max_episode_steps

        for step in range(max_steps):
            prefer = step >= max_steps - 1
            act_a = random_valid_action(env_a, rng_a, prefer_diagnosis=prefer)
            valid_b = masked_valid_action_ids(env_b)
            if prefer:
                specs = base_env(env_b).action_space_manager.action_map
                diag = [a for a in valid_b if specs[a].category == ActionCategory.DIAGNOSIS]
                act_b = rng_b.choice(diag) if diag else rng_b.choice(valid_b)
            else:
                non_diag = [
                    a
                    for a in valid_b
                    if base_env(env_b).action_space_manager.action_map[a].category
                    != ActionCategory.DIAGNOSIS
                ]
                act_b = rng_b.choice(non_diag or valid_b)

            if act_a not in set(get_valid_action_ids(env_a)):
                invalid_a += 1
            if act_b not in set(masked_valid_action_ids(env_b)):
                invalid_b += 1
            steps_a += 1
            steps_b += 1

            _, _, term_a, trunc_a, _ = env_a.step(act_a)
            _, _, term_b, trunc_b, _ = env_b.step(act_b)
            if term_a or trunc_a:
                break
            if term_b or trunc_b:
                break

        env_a.close()
        env_b.close()

    return {
        "episodes": n_episodes,
        "random_valid_invalid_steps": invalid_a,
        "masked_random_invalid_steps": invalid_b,
        "pass": invalid_a == 0 and invalid_b == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-states", type=int, default=500)
    parser.add_argument("--n-episodes", type=int, default=50)
    args = parser.parse_args()

    protocol = load_protocol()
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    (reports / "action_mask_audit.md").write_text(AUDIT_MD, encoding="utf-8")
    print(f"Wrote {reports / 'action_mask_audit.md'}")

    wrapper_result = validate_mask_wrapper(protocol, n_states=args.n_states)
    wrapper_lines = [
        "# Mask Wrapper Validation (Step 9.6-B)",
        "",
        f"- Episodes sampled: {wrapper_result['n_episodes_sampled']}",
        f"- State checks: {wrapper_result['samples_checked']}",
        f"- shape mismatch: {wrapper_result['shape_mismatch']}",
        f"- empty mask: {wrapper_result['empty_mask']}",
        f"- mask != get_valid_actions(): {wrapper_result['mask_mismatch_vs_get_valid_actions']}",
        f"- **PASS: {wrapper_result['pass']}**",
    ]
    path_b = reports / "mask_wrapper_validation.md"
    path_b.write_text("\n".join(wrapper_lines) + "\n", encoding="utf-8")
    print(f"Wrote {path_b} (pass={wrapper_result['pass']})")

    random_result = validate_masked_random(protocol, n_episodes=args.n_episodes)
    random_lines = [
        "# Masked Random Check (Step 9.6-C)",
        "",
        f"- Episodes: {random_result['episodes']}",
        f"- Random-Valid invalid steps: {random_result['random_valid_invalid_steps']}",
        f"- Mask-indexed random invalid steps: {random_result['masked_random_invalid_steps']}",
        f"- **PASS: {random_result['pass']}**",
    ]
    path_c = reports / "masked_random_check.md"
    path_c.write_text("\n".join(random_lines) + "\n", encoding="utf-8")
    print(f"Wrote {path_c} (pass={random_result['pass']})")


if __name__ == "__main__":
    main()
