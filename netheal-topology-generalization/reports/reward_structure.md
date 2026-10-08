# NetHeal Reward Structure (Step 9.5)

Source: `netheal/environment/rewards.py` + invalid action in `env.py`

{
  "network_size": 6,
  "step_penalty": -0.1,
  "correct_diagnosis_reward_scaled": 16.0,
  "incorrect_diagnosis_penalty_scaled": -16.0,
  "partial_device_reward_single_scaled": 4.800000000000001,
  "partial_device_reward_per_device_scaled": 2.4000000000000004,
  "invalid_action_penalty": -5.0,
  "diagnosis_terminates_episode": true,
  "max_step_truncation_reward": "no terminal bonus; last step still gets step_penalty",
  "scenarios": {
    "immediate_correct": 15.9,
    "immediate_wrong": -16.1,
    "explore_5_steps_then_correct": 15.5,
    "explore_10_steps_then_correct": 15.0,
    "explore_14_steps_then_correct": 14.6
  }
}
