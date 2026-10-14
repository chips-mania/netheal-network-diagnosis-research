"""Controlled fault injection at anchor locations for information audits."""
from __future__ import annotations

from typing import Dict, List, Optional

from netheal.faults.injector import FaultInfo, FaultType


ANCHOR = {
    "device": "edge_0",
    "link_source": "center",
    "link_dest": "edge_0",
}

STAR_CENTER = "center"
STAR_EDGES = ["edge_0", "edge_1", "edge_2", "edge_3", "edge_4"]
LOCATION_PROBE_EDGES = ["edge_0", "edge_1", "edge_2"]


def inject_controlled_fault(env, fault_type: FaultType) -> FaultInfo:
    """Clear random reset fault and inject at fixed anchor on current network."""
    inj = env.fault_injector
    inj.clear_all_faults()

    device = ANCHOR["device"]
    src, dst = ANCHOR["link_source"], ANCHOR["link_dest"]

    if fault_type == FaultType.LINK_FAILURE:
        fault = inj.inject_link_failure(src, dst)
    elif fault_type == FaultType.DEVICE_FAILURE:
        fault = inj.inject_device_failure(device)
    elif fault_type == FaultType.PERFORMANCE_DEGRADATION:
        fault = inj.inject_performance_degradation(src, dst)
    elif fault_type == FaultType.MISCONFIGURATION:
        fault = inj.inject_misconfiguration(device, dst)
    else:
        raise ValueError(f"Unsupported fault type: {fault_type}")

    env.ground_truth_fault = fault
    return fault


def anchor_description() -> Dict[str, str]:
    return {
        "link_failure": f"{ANCHOR['link_source']}->{ANCHOR['link_dest']}",
        "performance_degradation": f"{ANCHOR['link_source']}->{ANCHOR['link_dest']}",
        "device_failure": ANCHOR["device"],
        "misconfiguration": ANCHOR["device"],
    }


def inject_fault_at_location(env, fault_type: FaultType, location: str) -> FaultInfo:
    """Inject a fault at an explicit Star topology location."""
    inj = env.fault_injector
    inj.clear_all_faults()

    if fault_type == FaultType.DEVICE_FAILURE:
        fault = inj.inject_device_failure(location)
    elif fault_type == FaultType.MISCONFIGURATION:
        fault = inj.inject_misconfiguration(location, STAR_CENTER)
    elif fault_type == FaultType.PERFORMANCE_DEGRADATION:
        if "->" not in location:
            location = f"{STAR_CENTER}->{location}"
        src, dst = location.split("->", 1)
        fault = inj.inject_performance_degradation(src.strip(), dst.strip())
    elif fault_type == FaultType.LINK_FAILURE:
        if "->" not in location:
            location = f"{STAR_CENTER}->{location}"
        src, dst = location.split("->", 1)
        fault = inj.inject_link_failure(src.strip(), dst.strip())
    else:
        raise ValueError(f"Unsupported fault type: {fault_type}")

    env.ground_truth_fault = fault
    return fault


def default_probe_locations(fault_type: FaultType) -> List[str]:
    if fault_type in (FaultType.DEVICE_FAILURE, FaultType.MISCONFIGURATION):
        return list(LOCATION_PROBE_EDGES)
    return list(LOCATION_PROBE_EDGES)
