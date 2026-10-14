"""Symmetric type/location partial reward — Step 9.13 protocol variant."""
from __future__ import annotations

from typing import Dict, Tuple

from netheal.environment.actions import ActionCategory, ActionSpec
from netheal.environment.rewards import SparseRewardCalculator
from netheal.faults.injector import FaultInfo, FaultType


class SymmetricPartialRewardCalculator(SparseRewardCalculator):
    """Partial credit for correct type OR correct location (same scale), not location-only."""

    def calculate_reward(
        self, action_spec: ActionSpec, ground_truth: FaultInfo, network_size: int
    ) -> Tuple[float, Dict[str, float]]:
        reward = 0.0
        breakdown = {
            "step_penalty": 0.0,
            "diagnosis_reward": 0.0,
            "partial_device_reward": 0.0,
            "partial_type_reward": 0.0,
            "partial_location_reward": 0.0,
        }

        reward += self.step_penalty
        breakdown["step_penalty"] = self.step_penalty

        if action_spec.category != ActionCategory.DIAGNOSIS:
            return reward, breakdown

        diagnosed_fault = action_spec.action_type
        diagnosed_location = action_spec.parameters.get("location")
        scaling_multiplier = 1.0 + (network_size / self.scaling_factor)

        type_ok = diagnosed_fault == ground_truth.fault_type
        loc_ok = self._locations_match(
            diagnosed_fault, diagnosed_location, ground_truth.location
        )

        if type_ok and loc_ok:
            diagnosis_reward = self.correct_diagnosis_reward * scaling_multiplier
            reward += diagnosis_reward
            breakdown["diagnosis_reward"] = diagnosis_reward
            return reward, breakdown

        type_partial = self._type_only_partial(type_ok, network_size, scaling_multiplier)
        loc_partial = self._calculate_partial_device_reward(
            diagnosed_location, ground_truth.location, network_size
        )

        if type_ok and not loc_ok and type_partial > 0:
            reward += type_partial
            breakdown["partial_type_reward"] = type_partial
        elif not type_ok and loc_partial > 0:
            reward += loc_partial
            breakdown["partial_location_reward"] = loc_partial
        else:
            diagnosis_penalty = self.incorrect_diagnosis_penalty * scaling_multiplier
            reward += diagnosis_penalty
            breakdown["diagnosis_reward"] = diagnosis_penalty

        return reward, breakdown

    def _type_only_partial(
        self, type_ok: bool, network_size: int, scaling_multiplier: float
    ) -> float:
        if not type_ok:
            return 0.0
        return (
            self.correct_diagnosis_reward
            * self.single_device_reward_rate
            * scaling_multiplier
        )
