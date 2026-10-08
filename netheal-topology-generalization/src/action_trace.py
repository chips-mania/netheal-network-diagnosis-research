"""Action trace collection and aggregation for Step 9.5 PPO diagnosis."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from netheal.environment.actions import ActionCategory, ActionSpec
from netheal.evaluation.metrics import CompetitionEvaluator
from netheal.faults.injector import FaultType

from .env_utils import make_eval_env, metrics_env
from .metrics_utils import summarize_evaluator


TRACE_COLUMNS = [
    "episode_id",
    "seed",
    "agent",
    "fault_type",
    "fault_location",
    "step",
    "action_id",
    "action_category",
    "action_type",
    "target",
    "reward",
    "is_valid",
    "done",
    "diagnosis_correct",
]


@dataclass
class StepRecord:
    episode_id: int
    seed: int
    agent: str
    fault_type: str
    fault_location: str
    step: int
    action_id: int
    action_category: str
    action_type: str
    target: str
    reward: float
    is_valid: bool
    done: bool
    diagnosis_correct: Optional[bool]


@dataclass
class EpisodeSummary:
    episode_id: int
    seed: int
    agent: str
    fault_type: str
    fault_location: str
    steps: int
    success: bool
    predicted_type: Optional[str]
    predicted_location: Optional[str]
    invalid_action_count: int
    tool_actions_before_diagnosis: int
    first_action_is_diagnosis: bool
    coverage: float


ActionFn = Callable[[Any, Any, int, List[StepRecord]], int]


def _base_env(env):
    base = env.unwrapped
    while hasattr(base, "env"):
        base = base.env
    return base


def _format_target(spec: Optional[ActionSpec]) -> str:
    if spec is None:
        return ""
    params = spec.parameters or {}
    if "location" in params:
        return str(params["location"])
    if "source" in params and "destination" in params:
        return f"{params['source']}->{params['destination']}"
    if "device" in params:
        return str(params["device"])
    return ""


def _is_action_valid(base_env, action_id: int) -> bool:
    return action_id in set(base_env.get_valid_actions())


def _diagnosis_correct(info: Dict[str, Any], spec: Optional[ActionSpec]) -> Optional[bool]:
    if spec is None or spec.category != ActionCategory.DIAGNOSIS:
        return None
    gt = info.get("ground_truth_fault") or {}
    gt_type = gt.get("type")
    gt_loc = gt.get("location")
    pred_type = spec.action_type.value if hasattr(spec.action_type, "value") else str(spec.action_type)
    pred_loc = spec.parameters.get("location", "")
    if not gt_type or not gt_loc:
        return None
    if pred_type != gt_type:
        return False
    if pred_type in ("link_failure", "performance_degradation"):
        if "->" not in pred_loc or "->" not in gt_loc:
            return False
        pred_parts = sorted(p.strip() for p in pred_loc.split("->") if p.strip())
        gt_parts = sorted(p.strip() for p in gt_loc.split("->") if p.strip())
        return pred_parts == gt_parts
    return pred_loc == gt_loc


def _parse_action_spec(info: Dict[str, Any], base_env, action_id: int) -> Optional[ActionSpec]:
    raw = info.get("action_spec")
    if raw is not None:
        if isinstance(raw, ActionSpec):
            return raw
        if isinstance(raw, dict):
            category = ActionCategory(raw["category"])
            action_type_value = raw["action_type"]
            action_type = _resolve_action_type(category, action_type_value)
            return ActionSpec(
                category=category,
                action_type=action_type,
                parameters=dict(raw.get("parameters") or {}),
                description=str(raw.get("description", "")),
            )
    return base_env.action_space_manager.get_action_spec(action_id)


def _resolve_action_type(category: ActionCategory, value: str):
    if category == ActionCategory.TOPOLOGY_DISCOVERY:
        from netheal.environment.actions import TopologyAction

        return TopologyAction(value)
    if category == ActionCategory.DIAGNOSTIC:
        from netheal.environment.actions import DiagnosticAction

        return DiagnosticAction(value)
    return FaultType(value)


def _is_tool_action(spec: Optional[ActionSpec]) -> bool:
    if spec is None:
        return False
    return spec.category in (ActionCategory.TOPOLOGY_DISCOVERY, ActionCategory.DIAGNOSTIC)


def run_episode_with_policy(
    env,
    *,
    episode_id: int,
    seed: int,
    agent: str,
    choose_action: ActionFn,
) -> Tuple[List[StepRecord], EpisodeSummary]:
    base = _base_env(env)
    obs, info = env.reset(seed=seed)

    gt = info.get("ground_truth_fault") or {}
    fault_type = gt.get("type", "")
    fault_location = gt.get("location", "")

    records: List[StepRecord] = []
    invalid_count = 0
    tool_before_diag = 0
    saw_diagnosis = False
    predicted_type: Optional[str] = None
    predicted_location: Optional[str] = None

    step = 0
    while True:
        step += 1
        action_id = int(choose_action(env, obs, step, records))
        valid = _is_action_valid(base, action_id)
        if not valid:
            invalid_count += 1

        obs, reward, terminated, truncated, info = env.step(action_id)
        spec = _parse_action_spec(info, base, action_id)
        phase = info.get("factorized_phase")
        category = spec.category.value if spec else ("invalid" if not valid else "unknown")
        if spec and spec.category == ActionCategory.DIAGNOSIS and phase == "type_commit":
            category = "diagnosis_type"
        action_type = (
            spec.action_type.value if spec and hasattr(spec.action_type, "value") else ""
        )
        target = _format_target(spec)
        diag_correct = (
            None
            if phase == "type_commit"
            else _diagnosis_correct(info, spec)
        )

        if _is_tool_action(spec) and not saw_diagnosis:
            tool_before_diag += 1
        if spec and spec.category == ActionCategory.DIAGNOSIS and phase != "type_commit":
            saw_diagnosis = True
            predicted_type = action_type
            predicted_location = target

        records.append(
            StepRecord(
                episode_id=episode_id,
                seed=seed,
                agent=agent,
                fault_type=fault_type,
                fault_location=fault_location,
                step=step,
                action_id=action_id,
                action_category=category,
                action_type=action_type,
                target=target,
                reward=float(reward),
                is_valid=valid,
                done=bool(terminated or truncated),
                diagnosis_correct=diag_correct,
            )
        )

        if terminated or truncated:
            break

    metrics = metrics_env(env).last_episode_metrics
    summary = EpisodeSummary(
        episode_id=episode_id,
        seed=seed,
        agent=agent,
        fault_type=fault_type,
        fault_location=fault_location,
        steps=len(records),
        success=bool(metrics and metrics.diagnosis_success),
        predicted_type=predicted_type,
        predicted_location=predicted_location,
        invalid_action_count=invalid_count,
        tool_actions_before_diagnosis=tool_before_diag,
        first_action_is_diagnosis=bool(
            records and records[0].action_category == ActionCategory.DIAGNOSIS.value
        ),
        coverage=float(metrics.topology_coverage) if metrics else 0.0,
    )
    return records, summary


def run_agent_eval(
    protocol: dict,
    *,
    agent: str,
    n_episodes: int,
    choose_action: ActionFn,
    seed_base: int = 50000,
) -> Tuple[List[StepRecord], List[EpisodeSummary], Dict[str, Any]]:
    cfg = protocol["ppo_sanity"]
    faults = [FaultType(ft) for ft in protocol["fault_types"]]
    evaluator = CompetitionEvaluator()

    all_records: List[StepRecord] = []
    summaries: List[EpisodeSummary] = []

    for i in range(n_episodes):
        fault = faults[i % len(faults)]
        env = make_eval_env(cfg["topology_types"], fault_types=[fault], protocol=protocol)
        seed = seed_base + i
        records, summary = run_episode_with_policy(
            env,
            episode_id=i,
            seed=seed,
            agent=agent,
            choose_action=choose_action,
        )
        if metrics_env(env).last_episode_metrics is not None:
            evaluator.add_episode_metrics(metrics_env(env).last_episode_metrics)
        all_records.extend(records)
        summaries.append(summary)
        env.close()

    return all_records, summaries, summarize_evaluator(evaluator)


def write_trace_csv(path: Path, records: Sequence[StepRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "episode_id": rec.episode_id,
                    "seed": rec.seed,
                    "agent": rec.agent,
                    "fault_type": rec.fault_type,
                    "fault_location": rec.fault_location,
                    "step": rec.step,
                    "action_id": rec.action_id,
                    "action_category": rec.action_category,
                    "action_type": rec.action_type,
                    "target": rec.target,
                    "reward": rec.reward,
                    "is_valid": rec.is_valid,
                    "done": rec.done,
                    "diagnosis_correct": rec.diagnosis_correct,
                }
            )


def aggregate_behavior(
    summaries: Sequence[EpisodeSummary], records: Sequence[StepRecord]
) -> Dict[str, Any]:
    if not summaries:
        return {}

    first_diag_rate = float(np.mean([s.first_action_is_diagnosis for s in summaries]))
    invalid_rate = float(
        np.sum([s.invalid_action_count for s in summaries])
        / max(1, sum(s.steps for s in summaries))
    )
    tool_before = [s.tool_actions_before_diagnosis for s in summaries]
    success_tool_before = [s.tool_actions_before_diagnosis for s in summaries if s.success]

    action_type_counts: Counter = Counter()
    category_counts: Counter = Counter()
    for rec in records:
        if rec.action_category != ActionCategory.DIAGNOSIS.value:
            key = rec.action_type or rec.action_category
            action_type_counts[key] += 1
        category_counts[rec.action_category] += 1

    predicted_types = Counter(s.predicted_type for s in summaries if s.predicted_type)
    actual_types = Counter(s.fault_type for s in summaries)

    fault_dsr: Dict[str, float] = {}
    fault_groups: Dict[str, List[bool]] = defaultdict(list)
    for s in summaries:
        fault_groups[s.fault_type].append(s.success)
    for fault, outcomes in fault_groups.items():
        fault_dsr[fault] = float(np.mean(outcomes))

    confusion: Dict[str, Counter] = defaultdict(Counter)
    for s in summaries:
        if s.predicted_type:
            confusion[s.fault_type][s.predicted_type] += 1

    def _episode_uses(action_type: str) -> int:
        return sum(
            1
            for s in summaries
            if any(
                r.action_type == action_type and r.episode_id == s.episode_id for r in records
            )
        )

    n = len(summaries)
    return {
        "first_action_diagnosis_rate": first_diag_rate,
        "invalid_action_rate": invalid_rate,
        "avg_tool_actions_before_diagnosis": float(np.mean(tool_before)),
        "avg_tool_actions_before_diagnosis_success_only": (
            float(np.mean(success_tool_before)) if success_tool_before else None
        ),
        "scan_usage_rate": _episode_uses("scan_network") / n,
        "ping_usage_rate": _episode_uses("ping") / n,
        "traceroute_usage_rate": _episode_uses("traceroute") / n,
        "check_status_usage_rate": _episode_uses("check_status") / n,
        "check_interfaces_usage_rate": _episode_uses("check_interfaces") / n,
        "action_type_frequency": dict(action_type_counts.most_common()),
        "action_category_frequency": dict(category_counts.most_common()),
        "predicted_diagnosis_distribution": dict(predicted_types.most_common()),
        "actual_fault_distribution": dict(actual_types.most_common()),
        "fault_type_dsr": fault_dsr,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }


def format_confusion_matrix(confusion: Dict[str, Dict[str, int]], fault_types: Iterable[str]) -> List[str]:
    preds = sorted({p for row in confusion.values() for p in row.keys()})
    if not preds:
        return ["_(no predictions recorded)_"]
    header = ["actual \\ predicted"] + preds
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for actual in sorted(fault_types):
        row = confusion.get(actual, {})
        cells = [actual] + [str(row.get(p, 0)) for p in preds]
        lines.append("| " + " | ".join(cells) + " |")
    return lines
