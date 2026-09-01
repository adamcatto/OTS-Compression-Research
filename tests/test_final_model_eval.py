from pathlib import Path
from typing import Any, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ots_compression.core.experiments import ExperimentSpec
from ots_compression.nn_training.api import TaskDomain, TrainingTask
from ots_compression.nn_training.final_model_eval import evaluate_model_outputs


class _ClassificationTask(TrainingTask):
    name = "test.classification"
    domain = TaskDomain.OTHER

    def build_model(self, spec: ExperimentSpec) -> nn.Module:
        del spec
        return nn.Linear(2, 2, bias=False)

    def build_dataloaders(
        self, spec: ExperimentSpec, datasets_dir: Path
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        raise NotImplementedError

    def loss(
        self, model: nn.Module, batch: Any, spec: ExperimentSpec
    ) -> torch.Tensor:
        logits, targets = self.logits_and_targets(model, batch, spec)
        return F.cross_entropy(logits, targets)

    def logits_and_targets(
        self, model: nn.Module, batch: Any, spec: ExperimentSpec
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del spec
        inputs, targets = batch
        return model(inputs), targets


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        domain="test",
        task="test.classification",
        dataset={"name": "fixture"},
        model={"architecture": "linear"},
        optimizer={"name": "sgd", "lr": 0.1},
        training={"steps": 1, "device": "cpu"},
        seed=3,
    )


def test_final_model_output_metrics_detect_distribution_difference() -> None:
    reference = nn.Linear(2, 2, bias=False)
    lossy = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        reference.weight.copy_(torch.tensor([[2.0, -1.0], [-1.0, 2.0]]))
        lossy.weight.copy_(torch.tensor([[1.5, -0.5], [-0.5, 1.5]]))
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
            torch.tensor([0, 1, 0]),
        ),
        batch_size=2,
    )

    metrics, batches = evaluate_model_outputs(
        reference,
        lossy,
        loader,
        _ClassificationTask(),
        _spec(),
        device=torch.device("cpu"),
        evaluation_split="validation",
        max_batches=2,
    )

    assert metrics.evaluated_predictions == 3
    assert metrics.reference_to_lossy_kl > 0
    assert metrics.jensen_shannon_divergence > 0
    assert metrics.reference_to_lossy_cross_entropy > metrics.reference_to_lossy_kl
    assert metrics.top1_agreement == 1.0
    assert len(batches) == 2


def test_identical_final_models_have_zero_distribution_divergence() -> None:
    model = nn.Linear(2, 2, bias=False)
    loader = DataLoader(
        TensorDataset(torch.eye(2), torch.tensor([0, 1])), batch_size=2
    )

    metrics, _ = evaluate_model_outputs(
        model,
        model,
        loader,
        _ClassificationTask(),
        _spec(),
        device=torch.device("cpu"),
        evaluation_split="validation",
        max_batches=1,
    )

    assert metrics.reference_to_lossy_kl == pytest.approx(0.0, abs=1e-15)
    assert metrics.lossy_to_reference_kl == pytest.approx(0.0, abs=1e-15)
    assert metrics.jensen_shannon_divergence == pytest.approx(0.0, abs=1e-15)
    assert metrics.mean_absolute_probability_difference == 0.0
    assert metrics.top1_agreement == 1.0
