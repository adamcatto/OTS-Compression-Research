"""Built-in domain tasks and models."""

from ..api import TaskRegistry
from .nlp import CausalLanguageModelingTask
from .speech import SpeechCommandsTask
from .vision import VisionClassificationTask, VisionSegmentationTask


def default_task_registry() -> TaskRegistry:
    registry = TaskRegistry()
    registry.register(VisionClassificationTask())
    registry.register(VisionSegmentationTask())
    registry.register(CausalLanguageModelingTask())
    registry.register(SpeechCommandsTask())
    return registry


__all__ = [
    "CausalLanguageModelingTask",
    "SpeechCommandsTask",
    "VisionClassificationTask",
    "VisionSegmentationTask",
    "default_task_registry",
]

