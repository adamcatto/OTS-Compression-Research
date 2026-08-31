"""Project settings with environment and local ``.env`` overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


EXPERIMENTS_DIR_ENV = "OTS_EXPERIMENTS_DIR"
DATASETS_DIR_ENV = "OTS_DATASETS_DIR"


def _dotenv_value(path: Path, key: str) -> Optional[str]:
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        candidate, value = line.split("=", 1)
        if candidate.strip() != key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value
    return None


@dataclass(frozen=True)
class Settings:
    experiments_dir: Path
    datasets_dir: Path

    @classmethod
    def load(
        cls,
        *,
        environ: Optional[Mapping[str, str]] = None,
        dotenv_path: Path = Path(".env"),
    ) -> "Settings":
        values = os.environ if environ is None else environ
        experiments_dir = values.get(EXPERIMENTS_DIR_ENV)
        if experiments_dir is None:
            experiments_dir = _dotenv_value(Path(dotenv_path), EXPERIMENTS_DIR_ENV)
        if not experiments_dir:
            experiments_dir = "experiments"

        datasets_dir = values.get(DATASETS_DIR_ENV)
        if datasets_dir is None:
            datasets_dir = _dotenv_value(Path(dotenv_path), DATASETS_DIR_ENV)
        if not datasets_dir:
            datasets_dir = "data"

        return cls(
            experiments_dir=Path(experiments_dir).expanduser(),
            datasets_dir=Path(datasets_dir).expanduser(),
        )
