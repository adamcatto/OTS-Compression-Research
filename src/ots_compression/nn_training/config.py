"""Load an experiment specification from a checked-in JSON recipe."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.experiments import ExperimentSpec


def load_experiment_spec(path: Path) -> ExperimentSpec:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid experiment JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("experiment config must contain a JSON object")
    try:
        return ExperimentSpec(**value)
    except TypeError as exc:
        raise ValueError(f"invalid experiment fields in {path}: {exc}") from exc

