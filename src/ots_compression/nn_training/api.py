"""Integration points shared by vision, language, and audio training jobs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from ..core.gradient_trace import GradientTraceWriter
from ..core.experiments import ExperimentSpec


class TaskDomain(str, Enum):
    VISION = "vision"
    NLP = "nlp"
    AUDIO = "audio"
    OTHER = "other"


class GradientCaptureStage(str, Enum):
    """Point in the optimizer step represented by a gradient trace."""

    BACKWARD_OUTPUT = "backward_output"
    OPTIMIZER_INPUT = "optimizer_input"


class TrainingTask(ABC):
    """Small task adapter used by a future shared training loop.

    A task owns dataset/model construction and loss computation. Experiment
    configuration, optimization, and gradient recording remain common code.
    """

    name: str
    domain: TaskDomain

    @abstractmethod
    def build_model(self, spec: ExperimentSpec) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def build_dataloaders(
        self, spec: ExperimentSpec, datasets_dir: Path
    ) -> Tuple[Iterable[Any], Optional[Iterable[Any]]]:
        raise NotImplementedError

    @abstractmethod
    def loss(
        self, model: torch.nn.Module, batch: Any, spec: ExperimentSpec
    ) -> torch.Tensor:
        raise NotImplementedError


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: Dict[str, TrainingTask] = {}

    def register(self, task: TrainingTask) -> None:
        if task.name in self._tasks:
            raise ValueError(f"task is already registered: {task.name}")
        self._tasks[task.name] = task

    def get(self, name: str) -> TrainingTask:
        try:
            return self._tasks[name]
        except KeyError as exc:
            raise KeyError(f"unknown training task: {name}") from exc

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._tasks))


@dataclass
class GradientRecorder:
    """Thin PyTorch adapter around the append-only trace writer."""

    writer: GradientTraceWriter

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        experiment_id: str,
        capture_stage: GradientCaptureStage,
        durable: bool = False,
    ) -> "GradientRecorder":
        return cls(
            GradientTraceWriter(
                path,
                experiment_id=experiment_id,
                durable=durable,
                metadata={"gradient_capture": capture_stage.value},
            )
        )

    def record(
        self, step: int, gradients: Mapping[str, Optional[torch.Tensor]]
    ) -> None:
        self.writer.append(step, gradients)

    def record_model(self, step: int, model: torch.nn.Module) -> None:
        self.record(step, {name: parameter.grad for name, parameter in model.named_parameters()})

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> "GradientRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
