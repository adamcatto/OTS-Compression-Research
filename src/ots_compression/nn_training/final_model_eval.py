"""Compare reference and lossy-replay final-model output distributions."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Tuple

import torch

from ..core.experiments import ExperimentSpec
from ..core.settings import Settings
from .api import TrainingTask
from .runner import _move_to_device, _resolve_device, _seed_everything
from .tasks import default_task_registry


@dataclass(frozen=True)
class FinalModelOutputMetrics:
    evaluation_split: str
    evaluated_batches: int
    evaluated_predictions: int
    reference_label_cross_entropy: float
    lossy_label_cross_entropy: float
    label_cross_entropy_delta: float
    reference_to_lossy_cross_entropy: float
    reference_to_lossy_kl: float
    lossy_to_reference_kl: float
    jensen_shannon_divergence: float
    top1_agreement: float
    mean_absolute_logit_difference: float
    logit_relative_l2: float
    mean_absolute_probability_difference: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_model_outputs(
    reference_model: torch.nn.Module,
    lossy_model: torch.nn.Module,
    loader: Iterable[Any],
    task: TrainingTask,
    spec: ExperimentSpec,
    *,
    device: torch.device,
    evaluation_split: str,
    max_batches: int,
) -> Tuple[FinalModelOutputMetrics, list[dict]]:
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    reference_model.eval()
    lossy_model.eval()
    totals = {
        "predictions": 0,
        "logit_values": 0,
        "reference_label_ce": 0.0,
        "lossy_label_ce": 0.0,
        "reference_to_lossy_ce": 0.0,
        "reference_to_lossy_kl": 0.0,
        "lossy_to_reference_kl": 0.0,
        "js": 0.0,
        "agreement": 0,
        "absolute_logit_error": 0.0,
        "squared_logit_error": 0.0,
        "reference_logit_energy": 0.0,
        "absolute_probability_error": 0.0,
    }
    batch_rows = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = _move_to_device(batch, device)
            reference_logits, targets = task.logits_and_targets(
                reference_model, batch, spec
            )
            lossy_logits, lossy_targets = task.logits_and_targets(
                lossy_model, batch, spec
            )
            if (
                reference_logits.shape != lossy_logits.shape
                or not torch.equal(targets, lossy_targets)
                or reference_logits.ndim != 2
                or targets.ndim != 1
                or reference_logits.shape[0] != targets.numel()
            ):
                raise ValueError("final models produced incompatible outputs")
            # MPS does not implement float64. Move the comparatively small
            # evaluation outputs to CPU for stable high-precision reductions.
            reference_logits = (
                reference_logits.detach().to(device="cpu").to(torch.float64)
            )
            lossy_logits = lossy_logits.detach().to(device="cpu").to(torch.float64)
            targets = targets.detach().to(device="cpu").to(torch.long)
            reference_log = torch.log_softmax(reference_logits, dim=-1)
            lossy_log = torch.log_softmax(lossy_logits, dim=-1)
            reference_probability = reference_log.exp()
            lossy_probability = lossy_log.exp()
            mixture = 0.5 * (reference_probability + lossy_probability)
            mixture_log = mixture.clamp_min(1e-300).log()
            prediction_count = targets.numel()
            logit_count = reference_logits.numel()
            reference_label_ce = float(
                -reference_log.gather(1, targets[:, None]).sum()
            )
            lossy_label_ce = float(-lossy_log.gather(1, targets[:, None]).sum())
            reference_to_lossy_ce = float(
                -(reference_probability * lossy_log).sum()
            )
            reference_to_lossy_kl = float(
                (reference_probability * (reference_log - lossy_log)).sum()
            )
            lossy_to_reference_kl = float(
                (lossy_probability * (lossy_log - reference_log)).sum()
            )
            js = 0.5 * float(
                (reference_probability * (reference_log - mixture_log)).sum()
                + (lossy_probability * (lossy_log - mixture_log)).sum()
            )
            agreement = int(
                (reference_logits.argmax(dim=-1) == lossy_logits.argmax(dim=-1))
                .sum()
            )
            logit_error = reference_logits - lossy_logits
            absolute_logit_error = float(logit_error.abs().sum())
            squared_logit_error = float(logit_error.square().sum())
            reference_logit_energy = float(reference_logits.square().sum())
            absolute_probability_error = float(
                (reference_probability - lossy_probability).abs().sum()
            )
            totals["predictions"] += prediction_count
            totals["logit_values"] += logit_count
            totals["reference_label_ce"] += reference_label_ce
            totals["lossy_label_ce"] += lossy_label_ce
            totals["reference_to_lossy_ce"] += reference_to_lossy_ce
            totals["reference_to_lossy_kl"] += reference_to_lossy_kl
            totals["lossy_to_reference_kl"] += lossy_to_reference_kl
            totals["js"] += js
            totals["agreement"] += agreement
            totals["absolute_logit_error"] += absolute_logit_error
            totals["squared_logit_error"] += squared_logit_error
            totals["reference_logit_energy"] += reference_logit_energy
            totals["absolute_probability_error"] += absolute_probability_error
            batch_rows.append(
                {
                    "batch": batch_index,
                    "predictions": prediction_count,
                    "reference_label_cross_entropy": reference_label_ce
                    / prediction_count,
                    "lossy_label_cross_entropy": lossy_label_ce / prediction_count,
                    "reference_to_lossy_kl": reference_to_lossy_kl
                    / prediction_count,
                    "jensen_shannon_divergence": js / prediction_count,
                    "top1_agreement": agreement / prediction_count,
                }
            )
    predictions = int(totals["predictions"])
    logit_values = int(totals["logit_values"])
    if not predictions or not logit_values:
        raise ValueError("final-model evaluation loader produced no predictions")
    reference_ce = totals["reference_label_ce"] / predictions
    lossy_ce = totals["lossy_label_ce"] / predictions
    return (
        FinalModelOutputMetrics(
            evaluation_split=evaluation_split,
            evaluated_batches=len(batch_rows),
            evaluated_predictions=predictions,
            reference_label_cross_entropy=reference_ce,
            lossy_label_cross_entropy=lossy_ce,
            label_cross_entropy_delta=lossy_ce - reference_ce,
            reference_to_lossy_cross_entropy=totals["reference_to_lossy_ce"]
            / predictions,
            reference_to_lossy_kl=totals["reference_to_lossy_kl"] / predictions,
            lossy_to_reference_kl=totals["lossy_to_reference_kl"] / predictions,
            jensen_shannon_divergence=totals["js"] / predictions,
            top1_agreement=totals["agreement"] / predictions,
            mean_absolute_logit_difference=totals["absolute_logit_error"]
            / logit_values,
            logit_relative_l2=math.sqrt(
                totals["squared_logit_error"]
                / max(totals["reference_logit_energy"], 1e-30)
            ),
            mean_absolute_probability_difference=totals[
                "absolute_probability_error"
            ]
            / logit_values,
        ),
        batch_rows,
    )


def evaluate_saved_final_models(
    experiment_path: Path,
    replayed_model_path: Path,
    destination: Path,
    *,
    max_batches: int = 32,
) -> FinalModelOutputMetrics:
    experiment_path = Path(experiment_path)
    manifest = json.loads(
        (experiment_path / "manifest.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (experiment_path / "replay_contract.json").read_text(encoding="utf-8")
    )
    spec = ExperimentSpec(**manifest["spec"])
    settings = Settings.load()
    _seed_everything(spec.seed, bool(spec.training.get("deterministic", False)))
    task = default_task_registry().get(spec.task)
    device = _resolve_device(str(spec.training.get("device", "auto")))
    reference_model = task.build_model(spec)
    lossy_model = task.build_model(spec)
    reference_model.load_state_dict(
        torch.load(
            experiment_path / contract["final_model"]["path"],
            map_location="cpu",
            weights_only=True,
        )
    )
    lossy_model.load_state_dict(
        torch.load(Path(replayed_model_path), map_location="cpu", weights_only=True)
    )
    reference_model = reference_model.to(device)
    lossy_model = lossy_model.to(device)
    train_loader, validation_loader = task.build_dataloaders(
        spec, settings.datasets_dir
    )
    split = "validation" if validation_loader is not None else "train_probe"
    loader = validation_loader if validation_loader is not None else train_loader
    metrics, batch_rows = evaluate_model_outputs(
        reference_model,
        lossy_model,
        loader,
        task,
        spec,
        device=device,
        evaluation_split=split,
        max_batches=max_batches,
    )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "final_model_output_metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (destination / "final_model_output_batches.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in batch_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return metrics
