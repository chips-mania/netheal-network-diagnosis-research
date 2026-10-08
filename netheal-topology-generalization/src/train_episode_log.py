"""On-policy episode logger for PPO learn() — reward/mask unchanged."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym

from netheal.environment.actions import ActionCategory

from .action_mask import base_env
from .action_trace import _diagnosis_correct, _parse_action_spec
from .diagnosis_decomposition import location_match
from .pbrs import EvidenceBits, _gt, _tool_payload, ingest_tool


class TrainEpisodeLogWrapper(gym.Wrapper):
    """Write one JSONL row per finished training episode. No reward change."""

    train_episode_log = True

    def __init__(self, env: gym.Env, path: Path):
        super().__init__(env)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.episode_index = 0
        self.total_steps = 0
        self.rows: List[Dict[str, Any]] = []
        self._reset_episode()

    def _reset_episode(self) -> None:
        self.bits = EvidenceBits()
        self.phi_before_dx: Optional[int] = None
        self.bits_before_dx: Optional[dict] = None
        self.n_scan = 0
        self.n_diagnostic = 0
        self.n_diagnosis = 0
        self.submitted = False
        self.exact = False
        self.pred_type = ""
        self.pred_loc = ""
        self.fault_type = ""
        self.fault_loc = ""
        self.ep_steps = 0
        self.ep_reward = 0.0

    def reset(self, **kwargs):
        self._reset_episode()
        obs, info = self.env.reset(**kwargs)
        root = base_env(self.env)
        self.fault_type, self.fault_loc = _gt(root)
        return obs, info

    def step(self, action):
        root = base_env(self.env)
        spec = root.action_space_manager.get_action_spec(int(action))
        is_dx = spec is not None and spec.category == ActionCategory.DIAGNOSIS
        if is_dx and self.phi_before_dx is None:
            self.phi_before_dx = 0 if self.fault_type == "link_failure" else self.bits.phi
            self.bits_before_dx = self.bits.snapshot()

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.ep_steps += 1
        self.total_steps += 1
        self.ep_reward += float(reward)

        if spec is not None:
            if spec.category == ActionCategory.TOPOLOGY_DISCOVERY:
                self.n_scan += 1
            elif spec.category == ActionCategory.DIAGNOSTIC and not self.submitted:
                self.n_diagnostic += 1
            elif is_dx:
                self.n_diagnosis += 1
                self.submitted = True
                pred = spec.action_type
                self.pred_type = pred.value if hasattr(pred, "value") else str(pred)
                self.pred_loc = str((spec.parameters or {}).get("location") or "")
                parsed = _parse_action_spec(info, root, int(action))
                self.exact = bool(_diagnosis_correct(info, parsed))

        payload = _tool_payload(info.get("action_result"))
        if payload is not None:
            ingest_tool(self.bits, payload, self.fault_type, self.fault_loc)

        if terminated or truncated:
            self._flush_episode(bool(truncated and not terminated))
        return obs, reward, terminated, truncated, info

    def _flush_episode(self, timed_out: bool) -> None:
        bits = self.bits_before_dx or self.bits.snapshot()
        phi = self.phi_before_dx
        if phi is None:
            phi = 0 if self.fault_type == "link_failure" else self.bits.phi
        tool_use = self.n_diagnostic >= 1
        consistent = _consistent(
            phi,
            bits,
            self.fault_type,
            self.fault_loc,
            self.pred_type,
            self.pred_loc,
        )
        row = {
            "episode": self.episode_index,
            "t": self.total_steps,
            "fault_type": self.fault_type,
            "fault_location": self.fault_loc,
            "steps": self.ep_steps,
            "reward": self.ep_reward,
            "submitted": self.submitted,
            "timed_out": timed_out,
            "exact": self.exact,
            "tool_use": tool_use,
            "n_scan": self.n_scan,
            "n_diagnostic": self.n_diagnostic,
            "n_diagnosis": self.n_diagnosis,
            "phi_before_dx": int(phi),
            "type_bit": int(bits.get("type_bit", 0)),
            "loc_bit": int(bits.get("loc_bit", 0)),
            "exact_consistent": bool(self.exact and consistent),
            "predicted_type": self.pred_type,
            "predicted_location": self.pred_loc,
            "shortcut": self.n_scan >= 1 and self.n_diagnostic == 0 and self.submitted and self.ep_steps <= 2,
        }
        self._fh.write(json.dumps(row) + "\n")
        self.rows.append(row)
        self.episode_index += 1
        if self.episode_index % 500 == 0:
            self._fh.flush()
            print(f"train-log episodes={self.episode_index} t={self.total_steps} {summarize_rows(self.rows[-500:])}")

    def close(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        return self.env.close()


def _consistent(phi, bits, fault_type, fault_loc, pred_type, pred_loc) -> bool:
    if int(phi or 0) < 1:
        return False
    if bits.get("type_bit") and pred_type != fault_type:
        return False
    if bits.get("loc_bit") and not location_match(fault_loc or "", pred_loc or "", fault_type or ""):
        return False
    return True


def wrap_with_train_episode_log(env: gym.Env, path: Path) -> gym.Env:
    return TrainEpisodeLogWrapper(env, path)


def find_train_log(env) -> Optional[TrainEpisodeLogWrapper]:
    current = env
    while current is not None:
        if getattr(current, "train_episode_log", False):
            return current
        current = getattr(current, "env", None)
    return None


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(1, len(rows))
    tool = [r for r in rows if r["tool_use"]]
    nt = max(1, len(tool))
    phi = [r for r in rows if int(r.get("phi_before_dx", 0)) >= 1]
    na = max(1, len(phi))
    by: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        b = by[r.get("fault_type") or "unknown"]
        b["n"] += 1
        b["tool"] += int(r["tool_use"])
        b["exact"] += int(r["exact"])
        b["exact_tool"] += int(r["exact"] and r["tool_use"])
        b["exact_cons"] += int(r["exact_consistent"])
        b["shortcut"] += int(r.get("shortcut", False))
    return {
        "n": len(rows),
        "p_tool_use": sum(r["tool_use"] for r in rows) / n,
        "p_exact": sum(r["exact"] for r in rows) / n,
        "p_exact_and_tool": sum(r["exact"] and r["tool_use"] for r in rows) / n,
        "p_exact_given_tool": sum(r["exact"] and r["tool_use"] for r in tool) / nt if tool else 0.0,
        "p_exact_cons_given_tool": sum(r["exact_consistent"] for r in tool) / nt if tool else 0.0,
        "p_exact_cons_given_phi": sum(r["exact_consistent"] for r in phi) / na if phi else 0.0,
        "p_shortcut": sum(r.get("shortcut", False) for r in rows) / n,
        "per_fault": {k: dict(v) for k, v in sorted(by.items())},
    }
