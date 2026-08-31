"""Shared training runner and domain task implementations."""

from .api import (
    GradientCaptureStage,
    GradientRecorder,
    TaskDomain,
    TaskRegistry,
    TrainingTask,
)
from .config import load_experiment_spec
from .runner import TrainingResult, run_training

__all__ = [
    "GradientCaptureStage",
    "GradientRecorder",
    "TaskDomain",
    "TaskRegistry",
    "TrainingResult",
    "TrainingTask",
    "load_experiment_spec",
    "run_training",
]
