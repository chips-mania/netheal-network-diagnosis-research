# Engineering Note — Vanilla PPO Degenerate Policy (Step 9.5)

During initial Star-only sanity training (vanilla PPO + full discrete action space), we observed a **degenerate collapsed policy**: the agent selected the same terminal diagnosis action on step 1 in 100/100 evaluation episodes, with zero topology coverage.

- **Metrics:** DSR 4%, AvgSteps 1.0, Coverage 0 — identical to an immediate-random-diagnosis baseline.
- **Root cause (interface-level):** Vanilla PPO could select semantically invalid diagnosis actions not in `get_valid_actions()`; the environment still executed them and terminated the episode.
- **Fix:** Valid-action masking via `get_valid_actions()` + MaskablePPO (Step 9.6), not blind 100k training scale-up.

This is documented as an **environment–policy interface** finding, not a NetHeal benchmark bug claim.

See: `reports/ppo_diagnosis_step9_5.md`, `reports/action_mask_audit.md`.
