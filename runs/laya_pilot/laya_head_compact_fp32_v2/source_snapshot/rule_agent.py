"""GT-free rule agent: scan → disambiguate tools → diagnose from tool results only."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from netheal.environment.actions import ActionCategory, DiagnosticAction
from netheal.environment.observation import ConnectionStatus
from netheal.faults.injector import FaultType

from .action_mask import base_env
from .evidence_obs import structured_obs
from .scripted_expert import find_action, find_diagnosis, find_scan

LATENCY_MS = 10.0  # 10–20× normal can land ~5–100ms; 15 missed some Perf
STAR_CENTER = "center"


def _tool_payload(info: dict) -> dict:
    ar = info.get("action_result")
    if ar is None:
        return {}
    tr = getattr(ar, "result", None)
    if tr is None:
        return {
            "tool_name": getattr(ar, "tool_name", "") or "",
            "success": None,
            "data": {},
            "source": getattr(ar, "source", None),
            "destination": getattr(ar, "destination", None),
        }
    return {
        "tool_name": tr.tool_name,
        "success": tr.success,
        "data": dict(tr.data or {}),
        "source": getattr(ar, "source", None),
        "destination": getattr(ar, "destination", None),
    }


def _devices(root) -> List[str]:
    if root.network is None:
        return []
    return list(root.network.get_all_devices())


def _center(devices: List[str]) -> str:
    if STAR_CENTER in devices:
        return STAR_CENTER
    return devices[0] if devices else STAR_CENTER


def _expected_star_edges(devices: List[str]) -> Set[Tuple[str, str]]:
    c = _center(devices)
    edges = {d for d in devices if d != c}
    out: Set[Tuple[str, str]] = set()
    for e in edges:
        out.add((c, e))
        out.add((e, c))
    return out


def _connected_pairs(struct) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    if struct is None:
        return out
    matrix = struct.discovery_matrix
    for i in range(matrix.adjacency.shape[0]):
        for j in range(matrix.adjacency.shape[1]):
            if int(matrix.adjacency[i, j]) != ConnectionStatus.CONNECTED.value:
                continue
            src = matrix.reverse_map.get(i)
            dst = matrix.reverse_map.get(j)
            if src and dst:
                out.add((src, dst))
    return out


@dataclass
class RuleMemory:
    scanned: bool = False
    status: Dict[str, str] = field(default_factory=dict)
    ping: Dict[Tuple[str, str], dict] = field(default_factory=dict)
    ifaces: Dict[str, dict] = field(default_factory=dict)
    status_queue: List[str] = field(default_factory=list)
    ping_queue: List[Tuple[str, str]] = field(default_factory=list)
    iface_queue: List[str] = field(default_factory=list)
    missing: Set[Tuple[str, str]] = field(default_factory=set)
    conclusion: Optional[Tuple[str, str]] = None


class RuleAgent:
    """Stateful GT-free policy. Uses only obs / tool returns / topology type=star."""

    def __init__(self):
        self.mem = RuleMemory()

    def reset(self):
        self.mem = RuleMemory()

    def observe_step(self, info: dict):
        p = _tool_payload(info)
        tool = p.get("tool_name") or ""
        data = p.get("data") or {}
        if tool == "check_status":
            dev = data.get("device_id") or p.get("source")
            if dev:
                self.mem.status[str(dev)] = str(data.get("status", "unknown")).lower()
        elif tool == "ping":
            src, dst = p.get("source"), p.get("destination")
            if src and dst:
                lat = data.get("latency_ms")
                self.mem.ping[(str(src), str(dst))] = {
                    "success": bool(p.get("success")),
                    "latency_ms": float(lat) if lat is not None else None,
                    "error": str(data.get("error") or ""),
                }
        elif tool == "check_interfaces":
            dev = p.get("source") or data.get("device_id")
            if dev:
                self.mem.ifaces[str(dev)] = {
                    "success": bool(p.get("success")),
                    "interfaces": list(data.get("interfaces") or []),
                    "error": str(data.get("error") or ""),
                }

    def _try_conclude(self) -> Optional[Tuple[str, str]]:
        m = self.mem
        for dev, st in m.status.items():
            if st == "down":
                return FaultType.DEVICE_FAILURE.value, dev

        for (src, dst), pr in m.ping.items():
            if pr.get("success") and pr.get("latency_ms") is not None and pr["latency_ms"] >= LATENCY_MS:
                return FaultType.PERFORMANCE_DEGRADATION.value, f"{src}->{dst}"

        # Link vs misconfig from missing + interfaces / reverse missing
        for src, dst in sorted(m.missing):
            rev = (dst, src)
            asym = rev not in m.missing
            iface = m.ifaces.get(src)
            if iface and iface.get("success"):
                downs = [
                    str(i.get("destination"))
                    for i in iface.get("interfaces") or []
                    if str(i.get("status", "")).lower() == "down"
                ]
                if dst in downs and asym:
                    return FaultType.MISCONFIGURATION.value, src
                if dst in downs and not asym:
                    return FaultType.LINK_FAILURE.value, f"{src}->{dst}"
            if asym and src in m.status and m.status[src] == "up":
                # missing outbound only, device up → misconfig even before iface
                if src in m.ifaces or len(m.ifaces) > 0:
                    return FaultType.MISCONFIGURATION.value, src
            if not asym:
                # both directions missing; prefer link once endpoints up
                if m.status.get(src) == "up" and m.status.get(dst) == "up":
                    return FaultType.LINK_FAILURE.value, f"{src}->{dst}"

        # Ifaces alone: one down outbound
        for dev, iface in m.ifaces.items():
            if not iface.get("success"):
                # check_interfaces fail often means device down
                continue
            downs = [
                str(i.get("destination"))
                for i in iface.get("interfaces") or []
                if str(i.get("status", "")).lower() == "down"
            ]
            if len(downs) == 1:
                other = downs[0]
                if (other, dev) not in m.missing and (dev, other) in m.missing:
                    return FaultType.MISCONFIGURATION.value, dev
                if (other, dev) in m.missing:
                    return FaultType.LINK_FAILURE.value, f"{dev}->{other}"
            if len(downs) >= 1 and all((d, dev) in m.missing for d in downs):
                return FaultType.LINK_FAILURE.value, f"{dev}->{downs[0]}"
        return None

    def choose(self, env, obs, step, records) -> int:
        root = base_env(env)
        devices = _devices(root)
        c = _center(devices)
        m = self.mem

        if not m.scanned:
            m.scanned = True
            return find_scan(root)

        if step == 2 and not m.missing and not m.status_queue and not m.ping_queue:
            # First post-scan planning from discovery
            expected = _expected_star_edges(devices)
            connected = _connected_pairs(structured_obs(env))
            m.missing = {(a, b) for (a, b) in expected if (a, b) not in connected}
            if m.missing:
                ends: List[str] = []
                for a, b in sorted(m.missing):
                    for d in (a, b):
                        if d not in ends:
                            ends.append(d)
                m.status_queue = ends
                # Prefer check_interfaces on sources of missing edges
                srcs = []
                for a, _b in sorted(m.missing):
                    if a not in srcs:
                        srcs.append(a)
                m.iface_queue = srcs
            else:
                # Device or Perf: ping spokes first (fits in 15 steps), status only if needed
                edges = [d for d in devices if d != c]
                m.ping_queue = [(c, e) for e in sorted(edges)]
                m.status_queue = []  # filled after failed pings

        concluded = self._try_conclude()
        if concluded is not None:
            m.conclusion = concluded
            return find_diagnosis(root, concluded[0], concluded[1])

        while m.status_queue:
            dev = m.status_queue.pop(0)
            if dev in m.status:
                continue
            aid = find_action(
                root,
                lambda s, d=dev: s.action_type == DiagnosticAction.CHECK_STATUS
                and (s.parameters or {}).get("device") == d,
            )
            if aid is not None:
                return aid

        # Re-check after statuses filled
        concluded = self._try_conclude()
        if concluded is not None:
            m.conclusion = concluded
            return find_diagnosis(root, concluded[0], concluded[1])

        while m.iface_queue:
            dev = m.iface_queue.pop(0)
            if dev in m.ifaces:
                continue
            if m.status.get(dev) == "down":
                continue
            aid = find_action(
                root,
                lambda s, d=dev: s.action_type == DiagnosticAction.CHECK_INTERFACES
                and (s.parameters or {}).get("device") == d,
            )
            if aid is not None:
                return aid

        concluded = self._try_conclude()
        if concluded is not None:
            m.conclusion = concluded
            return find_diagnosis(root, concluded[0], concluded[1])

        while m.ping_queue:
            src, dst = m.ping_queue.pop(0)
            if (src, dst) in m.ping:
                continue
            aid = find_action(
                root,
                lambda s, a=src, b=dst: s.action_type == DiagnosticAction.PING
                and (s.parameters or {}).get("source") == a
                and (s.parameters or {}).get("destination") == b,
            )
            if aid is not None:
                return aid

        concluded = self._try_conclude()
        if concluded is not None:
            m.conclusion = concluded
            return find_diagnosis(root, concluded[0], concluded[1])

        # After spoke pings with no Perf hit: enqueue status for failed ping endpoints
        if not m.missing and not m.status_queue:
            suspects: List[str] = []
            for (src, dst), pr in m.ping.items():
                if pr.get("success"):
                    continue
                for d in (src, dst):
                    if d not in suspects:
                        suspects.append(d)
            if not suspects:
                suspects = sorted(devices, key=lambda d: (d != c, d))
            m.status_queue = [d for d in suspects if d not in m.status]

        while m.status_queue:
            dev = m.status_queue.pop(0)
            if dev in m.status:
                continue
            aid = find_action(
                root,
                lambda s, d=dev: s.action_type == DiagnosticAction.CHECK_STATUS
                and (s.parameters or {}).get("device") == d,
            )
            if aid is not None:
                return aid

        concluded = self._try_conclude()
        if concluded is not None:
            m.conclusion = concluded
            return find_diagnosis(root, concluded[0], concluded[1])

        # Fallback: if still missing and endpoints up → link on first missing pair
        if m.missing:
            src, dst = sorted(m.missing)[0]
            if (dst, src) in m.missing:
                return find_diagnosis(root, FaultType.LINK_FAILURE.value, f"{src}->{dst}")
            return find_diagnosis(root, FaultType.MISCONFIGURATION.value, src)

        # Relative latency: best ping among successes if absolute threshold missed
        ok = [(k, v) for k, v in m.ping.items() if v.get("success") and v.get("latency_ms") is not None]
        if ok:
            (src, dst), best = max(ok, key=lambda kv: kv[1]["latency_ms"])
            lats = [v["latency_ms"] for _, v in ok]
            if len(lats) >= 2:
                others = sorted(lats)[: max(1, len(lats) - 1)]
                med = others[len(others) // 2]
                if best["latency_ms"] >= max(LATENCY_MS, 5.0 * med):
                    return find_diagnosis(
                        root, FaultType.PERFORMANCE_DEGRADATION.value, f"{src}->{dst}"
                    )
            return find_diagnosis(root, FaultType.PERFORMANCE_DEGRADATION.value, f"{src}->{dst}")

        if devices:
            e = next((d for d in devices if d != c), c)
            return find_diagnosis(root, FaultType.PERFORMANCE_DEGRADATION.value, f"{c}->{e}")
        raise RuntimeError("rule agent stuck")


def make_rule_selector():
    agent = RuleAgent()

    def choose(env, obs, step, records):
        if step == 1:
            agent.reset()
        action = agent.choose(env, obs, step, records)
        return int(action)

    # attach for observe hook
    choose.agent = agent  # type: ignore[attr-defined]
    return choose


def run_episode_rule(env, *, episode_id: int, seed: int):
    """Like run_episode_with_policy but feeds tool results into the rule agent."""
    from .action_trace import (
        EpisodeSummary,
        StepRecord,
        _base_env,
        _diagnosis_correct,
        _format_target,
        _is_action_valid,
        _is_tool_action,
        _parse_action_spec,
    )
    from .env_utils import metrics_env
    from netheal.environment.actions import ActionCategory

    agent = RuleAgent()
    base = _base_env(env)
    obs, info = env.reset(seed=seed)
    gt = info.get("ground_truth_fault") or {}
    fault_type = gt.get("type", "")
    fault_location = gt.get("location", "")
    records: List[StepRecord] = []
    invalid_count = 0
    tool_before_diag = 0
    saw_diagnosis = False
    predicted_type = None
    predicted_location = None
    step = 0
    while True:
        step += 1
        action_id = int(agent.choose(env, obs, step, records))
        valid = _is_action_valid(base, action_id)
        if not valid:
            invalid_count += 1
        obs, reward, terminated, truncated, info = env.step(action_id)
        agent.observe_step(info)
        spec = _parse_action_spec(info, base, action_id)
        phase = info.get("factorized_phase")
        category = spec.category.value if spec else ("invalid" if not valid else "unknown")
        if spec and spec.category == ActionCategory.DIAGNOSIS and phase == "type_commit":
            category = "diagnosis_type"
        action_type = (
            spec.action_type.value if spec and hasattr(spec.action_type, "value") else ""
        )
        target = _format_target(spec)
        diag_correct = None if phase == "type_commit" else _diagnosis_correct(info, spec)
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
                agent="rule",
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
        agent="rule",
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
