"""Load and apply the configurable shot targets (config/targets.yaml)."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from e

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TARGETS_PATH = os.path.join(_HERE, "..", "config", "targets.yaml")


@dataclass
class Flag:
    metric: str
    value: float
    target: float | None
    band: tuple
    status: str          # "ok" | "low" | "high" | "na"
    confidence: str      # "high" | "medium" | "low"
    note: str = ""


def load_targets(path: str | None = None) -> dict:
    path = path or DEFAULT_TARGETS_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(metric: str, value, spec: dict, confidence: str = "high",
             note: str = "") -> Flag:
    """Compare a measured value against its target band -> Flag."""
    target = spec.get("target") if isinstance(spec, dict) else None
    band = spec.get("band", [None, None]) if isinstance(spec, dict) else [None, None]
    lo, hi = (band + [None, None])[:2]

    if value is None or (isinstance(value, float) and value != value):  # NaN
        return Flag(metric, value, target, tuple(band), "na", confidence,
                    note or "not measurable from this clip")

    status = "ok"
    if lo is not None and value < lo:
        status = "low"
    elif hi is not None and value > hi:
        status = "high"
    return Flag(metric, float(value), target, tuple(band), status, confidence, note)
