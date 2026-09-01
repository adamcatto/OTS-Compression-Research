"""Experiment identity, manifests, and filesystem layout."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .settings import Settings


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ExperimentSpec:
    """Everything expected to affect the generated gradient distribution.

    The main sections are deliberately mappings: model and optimizer ecosystems
    evolve faster than this schema, while their complete JSON configuration is
    still captured in the manifest and experiment fingerprint.
    """

    domain: str
    task: str
    dataset: Mapping[str, Any]
    model: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    training: Mapping[str, Any]
    seed: int
    scheduler: Mapping[str, Any] = field(default_factory=dict)
    initialization: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    tags: Tuple[str, ...] = ()
    notes: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.domain or not self.task:
            raise ValueError("domain and task must be non-empty")
        # Fail at experiment creation, not after a long training run.
        _canonical_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Experiment:
    """Resolved paths for one immutable experiment specification."""

    number: int
    path: Path
    spec: ExperimentSpec

    @property
    def experiment_id(self) -> str:
        return f"experiment_{self.number:03d}"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def gradients_path(self) -> Path:
        return self.path / "gradients.otsg"

    @property
    def baseline_results_path(self) -> Path:
        """Compatibility alias for the exact-codec result directory."""
        return self.path / "lossless"


class ExperimentStore:
    """Allocates ``experiments/experiment_NNN`` directories safely."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else Settings.load().experiments_dir

    def create(
        self, spec: ExperimentSpec, number: Optional[int] = None
    ) -> Experiment:
        self.root.mkdir(parents=True, exist_ok=True)

        candidates = [number] if number is not None else range(1, 1_000_000)
        for candidate in candidates:
            if candidate is None or candidate < 0:
                raise ValueError("experiment number must be non-negative")
            path = self.root / f"experiment_{candidate:03d}"
            try:
                path.mkdir()
            except FileExistsError:
                if number is not None:
                    raise
                continue

            experiment = Experiment(candidate, path, spec)
            manifest = {
                "experiment_id": experiment.experiment_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "runtime": {
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda,
                    "machine": platform.machine(),
                    "operating_system": platform.platform(),
                    "python_version": platform.python_version(),
                    "pytorch_version": torch.__version__,
                },
                "spec_fingerprint": spec.fingerprint,
                "spec": spec.to_dict(),
            }
            experiment.manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return experiment

        raise RuntimeError("no available experiment number")
