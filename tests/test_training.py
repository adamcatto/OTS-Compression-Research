import json
import sys
from pathlib import Path
from typing import Any, Optional, Tuple
from types import SimpleNamespace

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
from ots_compression.nn_training.runner import _resolve_device, run_training
from ots_compression.nn_training.tasks import default_task_registry
from ots_compression.nn_training.tasks.nlp import CausalLanguageModelingTask


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


def test_auto_device_prefers_mps_when_cuda_is_not_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _resolve_device("auto").type == "mps"


def test_explicit_mps_requires_an_available_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS was requested"):
        _resolve_device("mps")


def test_all_checked_in_training_recipes_build_one_to_ten_million_parameters() -> None:
    registry = default_task_registry()
    recipes = _checked_in_recipes()

    assert len(recipes) == 6
    for recipe in recipes:
        spec = load_experiment_spec(recipe)
        if spec.model["architecture"] == "huggingface_causal_lm":
            assert spec.model["pretrained_model"] == "roneneldan/TinyStories-1M"
            continue
        model = registry.get(spec.task).build_model(spec)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        assert 1_000_000 <= parameters <= 10_000_000, recipe


def test_every_recipe_architecture_completes_forward_and_backward() -> None:
    registry = default_task_registry()
    for recipe in _checked_in_recipes():
        spec = load_experiment_spec(recipe)
        if spec.model["architecture"] == "huggingface_causal_lm":
            continue
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
    assert GradientTraceReader(result.experiment.gradients_path).metadata[
        "gradient_capture"
    ] == "optimizer_input"
    assert (result.experiment.path / "training_metrics.jsonl").is_file()
    assert (result.experiment.path / "training_summary.json").is_file()


def test_capture_hook_can_record_before_or_after_gradient_clipping(
    tmp_path: Path,
) -> None:
    norms = {}
    for capture_stage in ("backward_output", "optimizer_input"):
        spec = ExperimentSpec(
            domain="test",
            task="test.regression",
            dataset={"name": "unit-test-fixture"},
            model={"architecture": "linear"},
            optimizer={"name": "sgd", "lr": 0.01},
            training={
                "steps": 1,
                "batch_size": 2,
                "device": "cpu",
                "precision": "fp32",
                "gradient_clip_norm": 0.01,
                "gradient_capture": capture_stage,
            },
            seed=11,
        )
        settings = Settings(
            experiments_dir=tmp_path / capture_stage,
            datasets_dir=tmp_path / "datasets",
        )
        result = run_training(spec, _RunnerFixtureTask(), settings=settings)
        gradients = next(
            GradientTraceReader(result.experiment.gradients_path).steps()
        ).gradients.values()
        norms[capture_stage] = sum(
            float(gradient.float().square().sum())
            for gradient in gradients
            if gradient is not None
        ) ** 0.5

    assert norms["backward_output"] > 0.01
    assert norms["optimizer_input"] == pytest.approx(0.01, rel=1e-4)


def test_meta_config_rejects_optimizer_options_from_the_wrong_variant(
    tmp_path: Path,
) -> None:
    recipe = Path("nn_training/configs/vision/cifar10_tiny_vit.json")
    value = json.loads(recipe.read_text(encoding="utf-8"))
    value["optimizer"]["momentum"] = 0.9
    invalid_recipe = tmp_path / "invalid.json"
    invalid_recipe.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="optimizer"):
        load_experiment_spec(
            invalid_recipe,
            meta_config_path=Path("nn_training/configs/experiment.schema.json"),
        )


def test_pretrained_causal_lm_config_and_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> nn.Module:
            calls.append((name, kwargs))
            return nn.Linear(2, 2)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=FakeAutoModel, AutoTokenizer=object),
    )
    spec = ExperimentSpec(
        domain="nlp",
        task="nlp.causal_lm",
        dataset={"name": "wikitext", "variant": "wikitext-2-raw-v1"},
        model={
            "architecture": "huggingface_causal_lm",
            "pretrained_model": "EleutherAI/pythia-14m",
            "sequence_length": 128,
            "revision": "main",
        },
        optimizer={"name": "adamw", "lr": 0.001},
        training={"steps": 1, "batch_size": 1},
        seed=0,
    )

    model = CausalLanguageModelingTask().build_model(spec)

    assert isinstance(model, nn.Linear)
    assert calls == [("EleutherAI/pythia-14m", {"revision": "main"})]
