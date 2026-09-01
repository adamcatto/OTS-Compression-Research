"""Replay a recorded (possibly lossy-decoded) gradient stream through AdamW."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..core.experiments import ExperimentSpec
from ..core.gradient_trace import GradientTraceReader
from ..core.settings import Settings
from .runner import _move_to_device, _next_batch, _optimizer, _resolve_device, _scheduler, _seed_everything
from .tasks import default_task_registry


@dataclass(frozen=True)
class ReplayMetrics:
    steps: int
    final_weight_relative_l2: float
    final_weight_cosine_similarity: float
    loss_mae: float
    loss_rmse: float
    loss_max_absolute_error: float

    def to_dict(self) -> dict:
        return asdict(self)


def replay_experiment(experiment_path: Path, gradient_trace: Path, destination: Path) -> ReplayMetrics:
    """Apply a trace to the saved initial model and compare it with the final model.

    Losses are evaluated before each update on a recreated data-loader sequence.
    They are a valid comparison when the task's loader is reproducible from the
    manifest seed; callers should treat them as a probe otherwise.
    """

    experiment_path, gradient_trace, destination = Path(experiment_path), Path(gradient_trace), Path(destination)
    manifest = json.loads((experiment_path / "manifest.json").read_text(encoding="utf-8"))
    spec = ExperimentSpec(**manifest["spec"])
    contract = json.loads((experiment_path / "replay_contract.json").read_text(encoding="utf-8"))
    settings = Settings.load()
    _seed_everything(spec.seed, bool(spec.training.get("deterministic", False)))
    task = default_task_registry().get(spec.task)
    device = _resolve_device(str(spec.training.get("device", "auto")))
    model = task.build_model(spec)
    initial = torch.load(experiment_path / contract["initial_model"]["path"], map_location="cpu", weights_only=True)
    model.load_state_dict(initial)
    model = model.to(device)
    loader, _ = task.build_dataloaders(spec, settings.datasets_dir)
    optimizer = _optimizer(model, spec.optimizer)
    scheduler = _scheduler(optimizer, spec.scheduler, int(spec.training["steps"]))
    parameters = dict(model.named_parameters())
    recorded_losses = [json.loads(line)["loss"] for line in (experiment_path / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    iterator = iter(loader)
    replay_losses = []
    model.train()
    for item in GradientTraceReader(gradient_trace).steps():
        batch, iterator = _next_batch(iterator, loader)
        with torch.no_grad():
            replay_losses.append(float(task.loss(model, _move_to_device(batch, device), spec)))
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in parameters.items():
            gradient = item.gradients.get(name)
            if gradient is None:
                parameter.grad = None
            else:
                parameter.grad = gradient.to(device=device, dtype=parameter.dtype)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    target = torch.load(experiment_path / contract["final_model"]["path"], map_location="cpu", weights_only=True)
    dot = source_energy = replay_energy = difference = 0.0
    for name, value in model.state_dict().items():
        replay_value, target_value = value.detach().cpu().to(torch.float64), target[name].to(torch.float64)
        dot += float(torch.sum(replay_value * target_value))
        replay_energy += float(torch.sum(replay_value.square()))
        source_energy += float(torch.sum(target_value.square()))
        difference += float(torch.sum((replay_value - target_value).square()))
    count = min(len(recorded_losses), len(replay_losses))
    deltas = [replay_losses[index] - recorded_losses[index] for index in range(count)]
    metrics = ReplayMetrics(
        steps=count,
        final_weight_relative_l2=math.sqrt(difference / max(source_energy, 1e-30)),
        final_weight_cosine_similarity=dot / max(math.sqrt(source_energy * replay_energy), 1e-30),
        loss_mae=sum(abs(value) for value in deltas) / max(count, 1),
        loss_rmse=math.sqrt(sum(value * value for value in deltas) / max(count, 1)),
        loss_max_absolute_error=max((abs(value) for value in deltas), default=0.0),
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "replay_metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (destination / "replay_loss_curve.jsonl").open("w", encoding="utf-8") as stream:
        for step, (original_loss, replay_loss) in enumerate(zip(recorded_losses, replay_losses)):
            stream.write(json.dumps({"step": step, "original_loss": original_loss, "replay_loss": replay_loss, "delta": replay_loss - original_loss}, sort_keys=True) + "\n")
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, destination / "replayed_final_model.pt")
    return metrics
