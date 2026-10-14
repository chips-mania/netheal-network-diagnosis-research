"""E1 shaping unit tests — must pass before PPO."""
from __future__ import annotations

from typing import Callable

from netheal.environment.actions import ActionCategory, DiagnosticAction, TopologyAction
from netheal.faults.injector import FaultType

from .action_mask import base_env
from .controlled_faults import ANCHOR, inject_controlled_fault, inject_fault_at_location
from .env_utils import make_base_env
from .pbrs import ALPHA, GAMMA, find_pbrs, wrap_with_pbrs
from .protocol import load_protocol


def _find(root, pred: Callable) -> int:
    for aid, spec in root.action_space_manager.action_map.items():
        if pred(spec):
            return aid
    raise RuntimeError("action not found")


def _scan(root) -> int:
    return _find(root, lambda s: s.action_type == TopologyAction.SCAN_NETWORK)


def _ping(root, src: str, dst: str) -> int:
    return _find(
        root,
        lambda s: s.action_type == DiagnosticAction.PING
        and s.parameters.get("source") == src
        and s.parameters.get("destination") == dst,
    )


def _ifaces(root, device: str) -> int:
    return _find(
        root,
        lambda s: s.action_type == DiagnosticAction.CHECK_INTERFACES
        and s.parameters.get("device") == device,
    )


def _diagnose(root, fault: FaultType, location: str) -> int:
    def pred(spec):
        if spec.category != ActionCategory.DIAGNOSIS:
            return False
        if spec.action_type != fault:
            return False
        return spec.parameters.get("location") == location

    return _find(root, pred)


def _setup(fault: FaultType):
    protocol = load_protocol()
    env = wrap_with_pbrs(make_base_env(["star"], fault_types=[fault], protocol=protocol))
    env.reset(seed=42)
    root = base_env(env)
    if fault == FaultType.MISCONFIGURATION:
        # ANCHOR link_dest == device would inject edge_0→edge_0. Block the spoke.
        inject_fault_at_location(root, fault, ANCHOR["device"])
    else:
        inject_controlled_fault(root, fault)
    return env, root, find_pbrs(env)


def _approx(got: float, expected: float, tol: float = 1e-6) -> None:
    if abs(got - expected) > tol:
        raise AssertionError(f"shaping {got} != {expected}")


def run_pbrs_unit_tests() -> dict:
    results = {}

    # 1: healthy ping after scan on device fault → Φ 0→0
    env, root, pbrs = _setup(FaultType.DEVICE_FAILURE)
    _, _, _, _, info = env.step(_scan(root))
    _, _, _, _, info = env.step(_ping(root, "edge_1", "edge_2"))
    _approx(info["pbrs_shaping"], 0.0)
    assert pbrs.bits.phi == 0, f"case1 phi={pbrs.bits.phi}"
    results["1"] = "ok"
    env.close()

    # 2: misconfig ping dest≠GT → type only, Φ 0→1
    env, root, pbrs = _setup(FaultType.MISCONFIGURATION)
    env.step(_scan(root))
    _, _, _, _, info = env.step(_ping(root, ANCHOR["device"], "center"))
    _approx(info["pbrs_shaping"], ALPHA * (GAMMA * 1 - 0))
    assert pbrs.bits.type_bit == 1 and pbrs.bits.loc_bit == 0, pbrs.bits.snapshot()
    results["2"] = "ok"

    # 3: same ping again — no new credit. Φ 1→1 is α(γ−1) ≈ −0.01, not a new plus.
    _, _, _, _, info = env.step(_ping(root, ANCHOR["device"], "center"))
    _approx(info["pbrs_shaping"], ALPHA * (GAMMA * 1 - 1))
    assert pbrs.bits.phi == 1
    if info["pbrs_shaping"] > 0.05:
        raise AssertionError("case 3 checkbox: unexpected positive shaping")
    results["3"] = "ok"

    # 4: interfaces on GT device → location, Φ 1→2
    _, _, _, _, info = env.step(_ifaces(root, ANCHOR["device"]))
    _approx(info["pbrs_shaping"], ALPHA * (GAMMA * 2 - 1))
    assert pbrs.bits.phi == 2, pbrs.bits.snapshot()
    results["4"] = "ok"

    # 5: diagnose → clawback -αΦ = -2
    loc = ANCHOR["device"]
    _, _, term, _, info = env.step(_diagnose(root, FaultType.MISCONFIGURATION, loc))
    _approx(info["pbrs_shaping"], -ALPHA * 2)
    assert term
    results["5"] = "ok"
    env.close()

    # 6: link scan then diagnose — all shaping 0
    env, root, pbrs = _setup(FaultType.LINK_FAILURE)
    _, _, _, _, info = env.step(_scan(root))
    _approx(info["pbrs_shaping"], 0.0)
    loc = f"{ANCHOR['link_source']}->{ANCHOR['link_dest']}"
    _, _, _, _, info = env.step(_diagnose(root, FaultType.LINK_FAILURE, loc))
    _approx(info["pbrs_shaping"], 0.0)
    assert pbrs.bits.phi == 0
    results["6"] = "ok"
    env.close()

    # 7: device ping(GT) → 0→2
    env, root, pbrs = _setup(FaultType.DEVICE_FAILURE)
    env.step(_scan(root))
    _, _, _, _, info = env.step(_ping(root, "center", ANCHOR["device"]))
    _approx(info["pbrs_shaping"], ALPHA * GAMMA * 2)
    assert pbrs.bits.phi == 2, pbrs.bits.snapshot()
    results["7"] = "ok"
    env.close()

    # 8: misconfig scan only → Φ=0
    env, root, pbrs = _setup(FaultType.MISCONFIGURATION)
    _, _, _, _, info = env.step(_scan(root))
    _approx(info["pbrs_shaping"], 0.0)
    assert pbrs.bits.phi == 0 and pbrs.bits.type_bit == 0
    results["8"] = "ok"
    env.close()

    return results
