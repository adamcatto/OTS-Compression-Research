"""Domain-independent optimizer loop that records every optimizer gradient."""

from __future__ import annotations

import json
import hashlib
import math
import random
import time
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import torch

from ..core.experiments import Experiment, ExperimentSpec, ExperimentStore
from ..core.settings import Settings
from ..compression.algorithms.ots_deltaq import OTSDeltaQCompressor
from .api import GradientCaptureStage, GradientRecorder, TrainingTask


def _state_dict_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_model_weights(model: torch.nn.Module, path: Path) -> str:
    """Save CPU tensors only, so replay never depends on an accelerator."""

    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(state, path)
    return _state_dict_sha256(path)


@dataclass(frozen=True)
class TrainingResult:
    experiment: Experiment
    completed_steps: int
    final_loss: float
    duration_seconds: float
    parameter_count: int
    device: str


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available in this PyTorch runtime")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "MPS was requested but is not available in this PyTorch runtime; "
            "install an Apple-Silicon PyTorch build and run on macOS 12.3 or later"
        )
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return type(value)((key, _move_to_device(item, device)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    values = dict(config)
    name = str(values.pop("name")).lower()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    try:
        optimizer_type = optimizers[name]
    except KeyError as exc:
        raise ValueError(f"unsupported optimizer: {name}") from exc
    if "betas" in values:
        values["betas"] = tuple(values["betas"])
    return optimizer_type(parameters, **values)


def _apply_initialization(
    model: torch.nn.Module, config: Mapping[str, Any]
) -> None:
    values = dict(config)
    source = str(values.pop("source", "random")).lower()
    if source == "random":
        if values:
            raise ValueError(
                "random initialization options belong in the model configuration: "
                + ", ".join(sorted(values))
            )
        return
    if source != "checkpoint":
        raise ValueError(f"unsupported initialization source: {source}")
    try:
        path = Path(values.pop("path"))
    except KeyError as exc:
        raise ValueError("checkpoint initialization requires a path") from exc
    state_dict_key = values.pop("state_dict_key", None)
    strict = bool(values.pop("strict", True))
    if values:
        raise ValueError(f"unknown checkpoint options: {sorted(values)}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if state_dict_key is not None:
        checkpoint = checkpoint[state_dict_key]
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint does not contain a state dictionary")
    model.load_state_dict(checkpoint, strict=strict)


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    values = dict(config)
    name = str(values.pop("name", "constant")).lower()
    if name == "constant":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=float(values.pop("min_lr", 0.0))
        )
    if name == "warmup_cosine":
        warmup_steps = int(values.pop("warmup_steps", 0))
        min_factor = float(values.pop("min_factor", 0.0))
        if values:
            raise ValueError(f"unknown warmup_cosine options: {sorted(values)}")

        def multiplier(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_factor + (1.0 - min_factor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    raise ValueError(f"unsupported scheduler: {name}")


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}
    try:
        dtype = dtypes[precision]
    except KeyError as exc:
        raise ValueError(f"unsupported precision: {precision}") from exc
    return torch.autocast(device_type=device.type, dtype=dtype)


def _next_batch(iterator: Iterator[Any], loader: Iterable[Any]) -> tuple[Any, Iterator[Any]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        try:
            return next(iterator), iterator
        except StopIteration as exc:
            raise ValueError("training dataloader is empty") from exc


def run_training(
    spec: ExperimentSpec,
    task: TrainingTask,
    *,
    store: Optional[ExperimentStore] = None,
    settings: Optional[Settings] = None,
    experiment_number: Optional[int] = None,
) -> TrainingResult:
    """Run one configured experiment and append gradients after every update."""

    if task.name != spec.task:
        raise ValueError(f"task {task.name!r} cannot run spec task {spec.task!r}")
    settings = settings or Settings.load()
    store = store or ExperimentStore(settings.experiments_dir)
    experiment = store.create(spec, number=experiment_number)

    training = dict(spec.training)
    total_steps = int(training["steps"])
    if total_steps <= 0:
        raise ValueError("training.steps must be positive")
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    deterministic = bool(training.get("deterministic", False))
    _seed_everything(spec.seed, deterministic)

    device = _resolve_device(str(training.get("device", "auto")))
    precision = str(training.get("precision", "fp32")).lower()
    try:
        capture_stage = GradientCaptureStage(
            str(
                training.get(
                    "gradient_capture", GradientCaptureStage.OPTIMIZER_INPUT.value
                )
            ).lower()
        )
    except ValueError as exc:
        choices = ", ".join(stage.value for stage in GradientCaptureStage)
        raise ValueError(f"gradient_capture must be one of: {choices}") from exc
    model = task.build_model(spec)
    _apply_initialization(model, spec.initialization)
    model = model.to(device)
    initial_weights_path = experiment.path / "initial_model.pt"
    initial_weights_sha256 = _save_model_weights(model, initial_weights_path)
    train_loader, _ = task.build_dataloaders(spec, settings.datasets_dir)
    optimizer = _optimizer(model, spec.optimizer)
    scheduler = _scheduler(optimizer, spec.scheduler, total_steps)
    use_scaler = device.type == "cuda" and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    iterator = iter(train_loader)
    metrics_path = experiment.path / "training_metrics.jsonl"
    clip_norm = training.get("gradient_clip_norm")

    started = time.perf_counter()
    completed_steps = 0
    final_loss = float("nan")
    model.train()
    online_config = training.get("online_compression")
    online_destination: Optional[Path] = None
    with ExitStack() as stack:
        recorder = stack.enter_context(
            GradientRecorder.open(
                experiment.gradients_path,
                experiment_id=experiment.experiment_id,
                capture_stage=capture_stage,
                durable=bool(training.get("durable_gradients", False)),
            )
        )
        online_writer = None
        if online_config is not None:
            online_values = dict(online_config)
            algorithm = str(online_values.pop("algorithm", "ots_deltaq_v1"))
            if algorithm not in {
                "ots_deltaq_v1",
                "ots_deltaq_zero_v1",
                "ots_deltaq_adaptive_predictor_v1",
                "ots_deltaq_outlier_v1",
                "ots_rank1_deltaq_e1",
                "ots_rank1_tracking_e2",
            }:
                raise ValueError(f"unsupported online compression algorithm: {algorithm}")
            save_predictions = bool(online_values.pop("save_predictions", True))
            compressor = OTSDeltaQCompressor(
                block_size=int(online_values.pop("block_size", 16_384)),
                relative_squared_error=float(
                    online_values.pop("relative_squared_error", 1e-4)
                ),
                prediction=(
                    "zero"
                    if algorithm == "ots_deltaq_zero_v1"
                    else (
                        "adaptive_zero_previous"
                        if algorithm == "ots_deltaq_adaptive_predictor_v1"
                        else "rank1_tensor"
                        if algorithm in {"ots_rank1_deltaq_e1", "ots_rank1_tracking_e2"}
                        else "previous_decoded_gradient"
                    )
                ),
                rank1_power_iterations=int(
                    online_values.pop("rank1_power_iterations", 1)
                ),
                rank1_warm_start=bool(
                    online_values.pop(
                        "rank1_warm_start",
                        algorithm == "ots_rank1_tracking_e2",
                    )
                ),
                outlier_fraction=float(
                    online_values.pop(
                        "outlier_fraction",
                        0.01 if algorithm == "ots_deltaq_outlier_v1" else 0.0,
                    )
                ),
            )
            if online_values:
                raise ValueError(
                    f"unknown online compression options: {sorted(online_values)}"
                )
            online_destination = (
                experiment.path / "lossy" / compressor.name / "online"
            )
            online_writer = stack.enter_context(
                compressor.open_online(
                    online_destination / "compressed.otsdq",
                    source_metadata={
                        "experiment_id": experiment.experiment_id,
                        "format": "ots-gradient-trace",
                        "gradient_capture": capture_stage.value,
                    },
                    predictions_path=(
                        online_destination / "predictions.otsg"
                        if save_predictions
                        else None
                    ),
                    prediction_metrics_path=(
                        online_destination / "prediction_metrics.jsonl"
                    ),
                    summary_path=online_destination / "online_summary.json",
                    durable=bool(training.get("durable_gradients", False)),
                )
            )
        metrics_stream = stack.enter_context(metrics_path.open("w", encoding="utf-8"))

        def record_gradients(step: int) -> None:
            gradients = {
                name: (
                    parameter.grad.detach().to(device="cpu")
                    if parameter.grad is not None
                    else None
                )
                for name, parameter in model.named_parameters()
            }
            recorder.record(step, gradients)
            if online_writer is not None:
                online_writer.append(step, gradients)

        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(accumulation_steps):
                batch, iterator = _next_batch(iterator, train_loader)
                batch = _move_to_device(batch, device)
                with _autocast_context(device, precision):
                    loss = task.loss(model, batch, spec)
                    backward_loss = loss / accumulation_steps
                if loss.ndim != 0:
                    raise ValueError("task loss must be a scalar tensor")
                accumulated_loss += float(loss.detach()) / accumulation_steps
                scaler.scale(backward_loss).backward()

            if capture_stage is GradientCaptureStage.BACKWARD_OUTPUT:
                record_gradients(step)
            if use_scaler:
                scaler.unscale_(optimizer)
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))

            if capture_stage is GradientCaptureStage.OPTIMIZER_INPUT:
                # These are the unscaled/clipped gradients consumed by the optimizer.
                record_gradients(step)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            final_loss = accumulated_loss
            completed_steps = step + 1
            metric = {
                "step": step,
                "loss": final_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            metrics_stream.write(json.dumps(metric, sort_keys=True) + "\n")
            metrics_stream.flush()

    duration_seconds = time.perf_counter() - started
    final_weights_path = experiment.path / "final_model.pt"
    final_weights_sha256 = _save_model_weights(model, final_weights_path)
    replay_contract = {
        "format_version": 1,
        "gradient_capture": capture_stage.value,
        "initial_model": {"path": initial_weights_path.name, "sha256": initial_weights_sha256},
        "final_model": {"path": final_weights_path.name, "sha256": final_weights_sha256},
        "optimizer": spec.optimizer,
        "scheduler": spec.scheduler,
        "parameter_order": [
            {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        "seed": spec.seed,
        "training": spec.training,
        "note": "Optimizer replay is deterministic for the recorded gradient stream; loss probes additionally require a reproducible batch manifest.",
    }
    (experiment.path / "replay_contract.json").write_text(
        json.dumps(replay_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = TrainingResult(
        experiment=experiment,
        completed_steps=completed_steps,
        final_loss=final_loss,
        duration_seconds=duration_seconds,
        parameter_count=parameter_count,
        device=str(device),
    )
    summary = {
        "completed_steps": result.completed_steps,
        "device": result.device,
        "duration_seconds": result.duration_seconds,
        "final_loss": result.final_loss,
        "gradient_capture": capture_stage.value,
        "parameter_count": result.parameter_count,
        "online_compression": (
            str(online_destination) if online_destination is not None else None
        ),
    }
    (experiment.path / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
