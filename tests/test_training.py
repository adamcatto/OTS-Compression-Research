import json
from pathlib import Path
from typing import Any, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ots_compression.core.experiments import ExperimentSpec, ExperimentStore
from ots_compression.core.gradient_trace import GradientTraceReader
from ots_compression.core.settings import Settings
from ots_compression.nn_training.api import TaskDomain, TrainingTask
from ots_compression.nn_training.config import load_experiment_spec
from ots_compression.nn_training.runner import run_training
from ots_compression.nn_training.tasks import default_task_registry


def _checked_in_recipes() -> list[Path]:
    return sorted(
        path
        for path in Path("nn_training/configs").rglob("*.json")
        if not path.name.endswith(".schema.json")
    )


class _RunnerFixtureTask(TrainingTask):
    name = "test.regression"
    domain = TaskDomain.OTHER

    def build_model(self, spec: ExperimentSpec) -> nn.Module:
        return nn.Linear(3, 1)

    def build_dataloaders(
        self, spec: ExperimentSpec, datasets_dir: Path
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        del spec, datasets_dir
        inputs = torch.tensor(
            [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
        )
        targets = inputs.sum(dim=1, keepdim=True)
        return DataLoader(TensorDataset(inputs, targets), batch_size=2), None

    def loss(
        self, model: nn.Module, batch: Any, spec: ExperimentSpec
    ) -> torch.Tensor:
        del spec
        inputs, targets = batch
        return F.mse_loss(model(inputs), targets)


def test_all_checked_in_training_recipes_build_one_to_ten_million_parameters() -> None:
    registry = default_task_registry()
    recipes = _checked_in_recipes()

    assert len(recipes) == 5
    for recipe in recipes:
        spec = load_experiment_spec(recipe)
        model = registry.get(spec.task).build_model(spec)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        assert 1_000_000 <= parameters <= 10_000_000, recipe


def test_every_recipe_architecture_completes_forward_and_backward() -> None:
    registry = default_task_registry()
    for recipe in _checked_in_recipes():
        spec = load_experiment_spec(recipe)
        task = registry.get(spec.task)
        model = task.build_model(spec)
        if spec.task == "vision.classification":
            batch = (torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
        elif spec.task == "vision.segmentation":
            batch = (
                torch.randn(1, 3, 32, 64),
                torch.zeros(1, 32, 64, dtype=torch.long),
            )
        elif spec.task == "nlp.causal_lm":
            batch = (
                torch.zeros(1, 8, dtype=torch.long),
                torch.ones(1, 8, dtype=torch.long),
            )
        else:
            batch = (torch.randn(1, 1_600), torch.zeros(1, dtype=torch.long))

        task.loss(model, batch, spec).backward()
        assert all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        ), recipe


def test_runner_records_every_optimizer_step(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        domain="test",
        task="test.regression",
        dataset={"name": "unit-test-fixture"},
        model={"architecture": "linear"},
        optimizer={"name": "sgd", "lr": 0.01},
        training={
            "steps": 3,
            "batch_size": 2,
            "device": "cpu",
            "precision": "fp32",
        },
        seed=11,
    )
    settings = Settings(
        experiments_dir=tmp_path / "experiments",
        datasets_dir=tmp_path / "datasets",
    )

    result = run_training(
        spec,
        _RunnerFixtureTask(),
        store=ExperimentStore(settings.experiments_dir),
        settings=settings,
    )

    steps = list(GradientTraceReader(result.experiment.gradients_path).steps())
    assert result.completed_steps == 3
    assert [record.step for record in steps] == [0, 1, 2]
    assert set(steps[0].gradients) == {"weight", "bias"}
    assert (result.experiment.path / "training_metrics.jsonl").is_file()
    assert (result.experiment.path / "training_summary.json").is_file()
