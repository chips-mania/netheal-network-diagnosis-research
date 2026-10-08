# Action Mask Audit (Step 9.6-A)

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
