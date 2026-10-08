"""Step 9.12 — Diagnosis action / reward factorization audit."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from netheal.environment.actions import ActionCategory
from netheal.environment.rewards import SparseRewardCalculator
from netheal.faults.injector import FaultInfo, FaultType

from .action_mask import base_env, valid_action_mask
from .controlled_faults import STAR_CENTER, inject_fault_at_location
from .env_utils import make_audit_env, make_masked_eval_env
from .protocol import load_protocol


@dataclass
class DiagnosisAction:
    action_id: int
    fault_type: str
    location: str


def _normalize_link(location: str) -> Tuple[str, ...]:
    if "->" not in location:
        return (location,)
    return tuple(sorted(p.strip() for p in location.split("->") if p.strip()))


def enumerate_diagnosis_actions(env) -> List[DiagnosisAction]:
    root = base_env(env)
    actions = []
    for aid, spec in root.action_space_manager.action_map.items():
        if spec.category != ActionCategory.DIAGNOSIS:
            continue
        ft = spec.action_type.value if hasattr(spec.action_type, "value") else str(spec.action_type)
        loc = spec.parameters.get("location", "")
        actions.append(DiagnosisAction(action_id=aid, fault_type=ft, location=loc))
    return actions


def audit_action_space(env) -> Dict[str, Any]:
    diag = enumerate_diagnosis_actions(env)
    devices = sorted(env.network.get_all_devices())
    n_devices = len(devices)
    n_links = n_devices * (n_devices - 1)

    by_type = Counter(d.fault_type for d in diag)
    device_locs = sorted({d.location for d in diag if "->" not in d.location})
    link_locs = sorted({d.location for d in diag if "->" in d.location})

    expected_device_slots = n_devices * 2  # device_failure + misconfiguration
    expected_link_slots = n_links * 2  # link + performance
    expected_total = expected_device_slots + expected_link_slots

    pairs = {(d.fault_type, d.location) for d in diag}
    is_cartesian = len(pairs) == len(diag)

    return {
        "n_devices": n_devices,
        "devices": devices,
        "n_directed_links": n_links,
        "total_action_space": env.action_space.n,
        "diagnosis_action_count": len(diag),
        "expected_diagnosis_count": expected_total,
        "is_fault_type_x_location_product": is_cartesian,
        "diagnosis_by_fault_type": dict(by_type),
        "unique_device_locations": len(device_locs),
        "unique_link_locations": len(link_locs),
        "mapping_table": [
            {"action_id": d.action_id, "fault_type": d.fault_type, "location": d.location}
            for d in sorted(diag, key=lambda x: x.action_id)
        ],
    }


def _find_diagnosis_action(env, fault_type: str, location: str) -> int:
    for d in enumerate_diagnosis_actions(env):
        if d.fault_type == fault_type and d.location == location:
            return d.action_id
    raise KeyError(f"No diagnosis action for {fault_type} @ {location}")


def _type_match(gt: FaultInfo, pred_type: str) -> bool:
    return gt.fault_type.value == pred_type


def _loc_match(gt: FaultInfo, pred_loc: str) -> bool:
    if gt.fault_type in (FaultType.LINK_FAILURE, FaultType.PERFORMANCE_DEGRADATION):
        return _normalize_link(gt.location) == _normalize_link(pred_loc)
    return gt.location == pred_loc


def audit_reward_grid(
    env,
    *,
    seed: int,
    fault_type: FaultType,
    fault_location: str,
    cases: Dict[str, Tuple[str, str]],
) -> Dict[str, Any]:
    """cases: label -> (pred_type, pred_location)"""
    env.reset(seed=seed)
    env.fault_injector.clear_all_faults()
    gt = inject_fault_at_location(env, fault_type, fault_location)
    network_size = len(env.network.get_all_devices())
    calc: SparseRewardCalculator = env.reward_calculator

    rows = []
    for label, (pred_type, pred_loc) in cases.items():
        aid = _find_diagnosis_action(env, pred_type, pred_loc)
        spec = env.action_space_manager.get_action_spec(aid)
        reward, breakdown = calc.calculate_reward(spec, gt, network_size)
        rows.append(
            {
                "case": label,
                "predicted_type": pred_type,
                "predicted_location": pred_loc,
                "action_id": aid,
                "type_correct": _type_match(gt, pred_type),
                "location_correct": _loc_match(gt, pred_loc),
                "exact_correct": _type_match(gt, pred_type) and _loc_match(gt, pred_loc),
                "total_reward": reward,
                "breakdown": breakdown,
            }
        )

    return {
        "ground_truth_type": gt.fault_type.value,
        "ground_truth_location": gt.location,
        "network_size": network_size,
        "scaling_multiplier": 1.0 + (network_size / calc.scaling_factor),
        "full_reward_base": calc.correct_diagnosis_reward,
        "penalty_base": calc.incorrect_diagnosis_penalty,
        "cases": rows,
    }


def build_standard_reward_cases(fault_type: FaultType, anchor: str) -> Dict[str, Tuple[str, str]]:
    """Four-way grid: TT/TN/NT/NN relative to injected fault at anchor."""
    if fault_type == FaultType.DEVICE_FAILURE:
        wrong_loc = "edge_1" if anchor != "edge_1" else "edge_2"
        return {
            "type_ok_loc_ok": (FaultType.DEVICE_FAILURE.value, anchor),
            "type_ok_loc_wrong": (FaultType.DEVICE_FAILURE.value, wrong_loc),
            "type_wrong_loc_ok": (FaultType.PERFORMANCE_DEGRADATION.value, f"{STAR_CENTER}->{anchor}"),
            "both_wrong": (FaultType.LINK_FAILURE.value, f"{STAR_CENTER}->edge_3"),
        }
    if fault_type == FaultType.PERFORMANCE_DEGRADATION:
        loc = f"{STAR_CENTER}->{anchor}"
        wrong_link = f"{STAR_CENTER}->edge_0" if anchor != "edge_0" else f"{STAR_CENTER}->edge_2"
        return {
            "type_ok_loc_ok": (FaultType.PERFORMANCE_DEGRADATION.value, loc),
            "type_ok_loc_wrong": (FaultType.PERFORMANCE_DEGRADATION.value, wrong_link),
            "type_wrong_loc_ok": (FaultType.LINK_FAILURE.value, loc),
            "both_wrong": (FaultType.DEVICE_FAILURE.value, "edge_3"),
        }
    if fault_type == FaultType.MISCONFIGURATION:
        wrong_loc = "edge_1" if anchor != "edge_1" else "edge_2"
        return {
            "type_ok_loc_ok": (FaultType.MISCONFIGURATION.value, anchor),
            "type_ok_loc_wrong": (FaultType.MISCONFIGURATION.value, wrong_loc),
            "type_wrong_loc_ok": (FaultType.LINK_FAILURE.value, f"{anchor}->{STAR_CENTER}"),
            "both_wrong": (FaultType.DEVICE_FAILURE.value, "edge_3"),
        }
    if fault_type == FaultType.LINK_FAILURE:
        loc = f"{STAR_CENTER}->{anchor}"
        wrong_link = f"{STAR_CENTER}->edge_1" if anchor != "edge_1" else f"{STAR_CENTER}->edge_2"
        return {
            "type_ok_loc_ok": (FaultType.LINK_FAILURE.value, loc),
            "type_ok_loc_wrong": (FaultType.LINK_FAILURE.value, wrong_link),
            "type_wrong_loc_ok": (FaultType.PERFORMANCE_DEGRADATION.value, loc),
            "both_wrong": (FaultType.DEVICE_FAILURE.value, "edge_3"),
        }
    raise ValueError(fault_type)


def audit_post_scan_valid_actions(
    env,
    *,
    seed: int,
    fault_type: Optional[FaultType] = None,
    fault_location: Optional[str] = None,
) -> Dict[str, Any]:
    from netheal.environment.actions import TopologyAction

    env.reset(seed=seed)
    env.fault_injector.clear_all_faults()
    if fault_type and fault_location:
        inject_fault_at_location(env, fault_type, fault_location)

    pre_valid = env.get_valid_actions()
    pre_mask = valid_action_mask(env)
    pre_by_cat = _count_valid_by_category(env, pre_valid)

    scan_id = None
    for aid, spec in env.action_space_manager.action_map.items():
        if spec.action_type == TopologyAction.SCAN_NETWORK:
            scan_id = aid
            break
    env.step(scan_id)

    post_valid = env.get_valid_actions()
    post_mask = valid_action_mask(env)
    post_by_cat = _count_valid_by_category(env, post_valid)

    post_diag = [
        d for d in enumerate_diagnosis_actions(env) if d.action_id in post_valid
    ]

    return {
        "pre_scan_valid_total": len(pre_valid),
        "post_scan_valid_total": len(post_valid),
        "pre_scan_by_category": pre_by_cat,
        "post_scan_by_category": post_by_cat,
        "post_scan_diagnosis_count": len(post_diag),
        "post_scan_diagnosis_by_type": dict(Counter(d.fault_type for d in post_diag)),
        "post_scan_non_diagnosis": post_by_cat.get("diagnosis", 0),
    }


def _count_valid_by_category(env, valid_ids: List[int]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for aid in valid_ids:
        spec = env.action_space_manager.get_action_spec(aid)
        if spec:
            counts[spec.category.value] += 1
    return dict(counts)


def _diagnosis_prob_marginals(model, env, obs: dict) -> Dict[str, Any]:
    import torch

    masks = valid_action_mask(env)
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(
            obs_tensor,
            action_masks=torch.as_tensor(np.array([masks])),
        )
        probs = dist.distribution.probs.cpu().numpy().flatten()

    diag_actions = [d for d in enumerate_diagnosis_actions(env) if masks[d.action_id]]
    type_probs: Dict[str, float] = defaultdict(float)
    loc_probs: Dict[str, float] = defaultdict(float)
    for d in diag_actions:
        p = float(probs[d.action_id])
        type_probs[d.fault_type] += p
        loc_probs[d.location] += p

    top_actions = sorted(
        ((d.action_id, d.fault_type, d.location, float(probs[d.action_id])) for d in diag_actions),
        key=lambda x: x[3],
        reverse=True,
    )[:8]

    return {
        "type_marginal": dict(type_probs),
        "location_marginal": dict(sorted(loc_probs.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "top_diagnosis_actions": [
            {"action_id": a, "fault_type": t, "location": l, "prob": p} for a, t, l, p in top_actions
        ],
        "diagnosis_prob_mass": sum(float(probs[d.action_id]) for d in diag_actions),
        "valid_diagnosis_count": len(diag_actions),
    }


def audit_ppo_diagnosis_marginals(
    model,
    protocol: dict,
    fault_type: FaultType,
    *,
    n_episodes: int = 20,
    seed_base: int = 90000,
) -> Dict[str, Any]:
    from netheal.environment.actions import TopologyAction

    episodes = []
    for i in range(n_episodes):
        seed = seed_base + i
        env = make_masked_eval_env(["star"], fault_types=[fault_type], protocol=protocol)
        base = env.unwrapped
        while hasattr(base, "env"):
            base = base.env

        obs, info = env.reset(seed=seed)
        gt = info.get("ground_truth_fault") or {}
        scan_id = next(
            aid
            for aid, spec in base.action_space_manager.action_map.items()
            if spec.action_type == TopologyAction.SCAN_NETWORK
        )
        obs, _, _, _, _ = env.step(scan_id)

        marginals = _diagnosis_prob_marginals(model, env, obs)
        gt_type = gt.get("type", "")
        gt_loc = gt.get("location", "")
        type_p = marginals["type_marginal"].get(gt_type, 0.0)
        loc_p = marginals["location_marginal"].get(gt_loc, 0.0)
        # link locations may appear reversed in action space
        if loc_p == 0.0 and "->" in gt_loc:
            rev = "->".join(reversed(gt_loc.split("->", 1)))
            loc_p = marginals["location_marginal"].get(rev, 0.0)

        episodes.append(
            {
                "episode_id": i,
                "seed": seed,
                "ground_truth_type": gt_type,
                "ground_truth_location": gt_loc,
                "p_correct_type": type_p,
                "p_correct_location": loc_p,
                **marginals,
            }
        )
        env.close()

    return {
        "fault_type": fault_type.value,
        "n_episodes": n_episodes,
        "mean_p_correct_type": float(np.mean([e["p_correct_type"] for e in episodes])),
        "mean_p_correct_location": float(np.mean([e["p_correct_location"] for e in episodes])),
        "episodes": episodes,
    }


def run_full_audit(protocol: dict, *, network_seed: int = 42, anchor: str = "edge_0") -> Dict[str, Any]:
    env = make_audit_env(["star"], protocol=protocol)
    env.reset(seed=network_seed)
    action_space = audit_action_space(env)

    reward_grids = {}
    for ft in (
        FaultType.LINK_FAILURE,
        FaultType.DEVICE_FAILURE,
        FaultType.PERFORMANCE_DEGRADATION,
        FaultType.MISCONFIGURATION,
    ):
        if ft in (FaultType.LINK_FAILURE, FaultType.PERFORMANCE_DEGRADATION):
            loc = f"{STAR_CENTER}->{anchor}"
        else:
            loc = anchor
        reward_grids[ft.value] = audit_reward_grid(
            env,
            seed=network_seed,
            fault_type=ft,
            fault_location=loc,
            cases=build_standard_reward_cases(ft, anchor),
        )

    post_scan = audit_post_scan_valid_actions(env, seed=network_seed)
    post_scan_m = audit_post_scan_valid_actions(
        env,
        seed=network_seed,
        fault_type=FaultType.MISCONFIGURATION,
        fault_location=anchor,
    )
    env.close()

    return {
        "network_seed": network_seed,
        "anchor": anchor,
        "action_space": action_space,
        "reward_grids": reward_grids,
        "post_scan_valid_actions": post_scan,
        "post_scan_valid_actions_misconfig": post_scan_m,
    }
