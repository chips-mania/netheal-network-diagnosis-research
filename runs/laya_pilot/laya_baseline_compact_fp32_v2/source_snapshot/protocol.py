"""Load experiment protocol from YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
NETHEAL_ROOT = Path(__file__).resolve().parents[2] / "netheal-ai-agent-benchmark"


def load_yaml(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_protocol() -> Dict[str, Any]:
    return load_yaml("protocol.yaml")


def load_eval_suite() -> Dict[str, Any]:
    return load_yaml("eval_suite.yaml")


def fault_type_enum(name: str):
    from netheal.faults.injector import FaultType

    return FaultType(name)
