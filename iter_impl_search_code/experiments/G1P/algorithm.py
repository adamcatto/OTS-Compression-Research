"""G1P: reconstruct gradients by replaying the recorded training procedure.

This is deliberately a side-information control, not a standalone gradient
codec.  The archive contains the initial weights, an explicit batch schedule,
the experiment contract, and this decoder source.  The dataset and repository
at the recorded revision remain external decoder inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import tempfile
import time
import zipfile
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch

from iter_impl_search_code.experiments.G0.algorithm import _energy, _read_selected_steps
from ots_compression.core.experiments import ExperimentSpec
from ots_compression.nn_training.runner import (
    _autocast_context,
    _move_to_device,
    _optimizer,
    _resolve_device,
    _scheduler,
    _seed_everything,
)
from ots_compression.nn_training.tasks import default_task_registry


_FORMAT = "ots-procedural-gradient-transcript-g1p"
_SCHEDULE_NAME = "batch_indices.u32.zlib"
_CONTRACT_NAME = "contract.json"
_INITIAL_NAME = "initial_model.pt"
_SOURCE_NAME = "g1p_decoder.py"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _spec_from_manifest(manifest: Mapping[str, Any]) -> ExperimentSpec:
    values = dict(manifest["spec"])
    values["tags"] = tuple(values.get("tags", ()))
    return ExperimentSpec(**values)


def _dataset_sha256(dataset: Any) -> str:
    tokens = dataset.tokens.detach().cpu().contiguous().numpy()
    return _sha256_bytes(tokens.tobytes(order="C"))


def sampler_schedule(dataset_size: int, batch_size: int, steps: int, seed: int) -> np.ndarray:
    """Mirror the single-worker DataLoader schedule, but store its result.

    DataLoader consumes one int64 from its generator for the worker base seed
    before RandomSampler asks for the first permutation.  The experiment has
    enough blocks that the requested 1,000 steps stay inside one epoch.
    """

    count = batch_size * steps
    if count > dataset_size:
        raise ValueError("G1P currently requires the sampled prefix to fit in one epoch")
    generator = torch.Generator().manual_seed(seed)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    indices = torch.randperm(dataset_size, generator=generator)[:count]
    return indices.reshape(steps, batch_size).to(torch.int64).numpy().astype("<u4")


def _zip_entry(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_transcript(
    experiment: Path,
    destination: Path,
    *,
    manifest: Mapping[str, Any],
    dataset: Any,
    schedule: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    initial = experiment / _INITIAL_NAME
    schedule_raw = np.asarray(schedule, dtype="<u4").tobytes(order="C")
    schedule_payload = zlib.compress(schedule_raw, level=9)
    source = Path(__file__).read_bytes()
    repository = Path(__file__).resolve().parents[3]
    spec = manifest["spec"]
    contract: dict[str, Any] = {
        "format": _FORMAT,
        "format_version": 1,
        "external_side_information": [
            "dataset content matching dataset_tokens_sha256",
            "OTS repository matching repository_commit",
            "compatible Python/PyTorch execution environment",
        ],
        "repository_commit": _git_commit(repository),
        "decoder_source_sha256": _sha256_bytes(source),
        "initial_model_sha256": _sha256_file(initial),
        "dataset": spec["dataset"],
        "dataset_tokens": int(dataset.tokens.numel()),
        "dataset_tokens_sha256": _dataset_sha256(dataset),
        "batch_schedule": {
            "batch_size": int(schedule.shape[1]),
            "dtype": "uint32-little-endian",
            "labels": "derived from the next token in the external dataset",
            "shape": list(schedule.shape),
            "compressed_sha256": _sha256_bytes(schedule_payload),
        },
        "model": spec["model"],
        "optimizer": spec["optimizer"],
        "scheduler": spec["scheduler"],
        "training": {
            key: value
            for key, value in spec["training"].items()
            if key != "online_compression"
        },
        "seed": int(spec["seed"]),
        "runtime": {
            "device": str(device),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
        },
        "random_payload_bytes": 0,
        "label_payload_bytes": 0,
        "random_access": "sequential from the included initial state; optional keyframes are costed separately",
    }
    contract_payload = json.dumps(contract, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        _zip_entry(archive, _CONTRACT_NAME, contract_payload)
        _zip_entry(archive, _INITIAL_NAME, initial.read_bytes())
        _zip_entry(archive, _SCHEDULE_NAME, schedule_payload)
        _zip_entry(archive, _SOURCE_NAME, source)
    return contract


def read_transcript(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor], np.ndarray]:
    with zipfile.ZipFile(path, "r") as archive:
        contract_payload = archive.read(_CONTRACT_NAME)
        contract = json.loads(contract_payload)
        if contract.get("format") != _FORMAT or contract.get("format_version") != 1:
            raise ValueError("unsupported G1P transcript")
        source = archive.read(_SOURCE_NAME)
        if _sha256_bytes(source) != contract["decoder_source_sha256"]:
            raise ValueError("decoder source hash mismatch")
        schedule_payload = archive.read(_SCHEDULE_NAME)
        if _sha256_bytes(schedule_payload) != contract["batch_schedule"]["compressed_sha256"]:
            raise ValueError("batch schedule hash mismatch")
        schedule_raw = zlib.decompress(schedule_payload)
        shape = tuple(int(value) for value in contract["batch_schedule"]["shape"])
        schedule = np.frombuffer(schedule_raw, dtype="<u4").reshape(shape).copy()
        with tempfile.NamedTemporaryFile(suffix=".pt") as temporary:
            temporary.write(archive.read(_INITIAL_NAME))
            temporary.flush()
            if _sha256_file(Path(temporary.name)) != contract["initial_model_sha256"]:
                raise ValueError("initial model hash mismatch")
            state = torch.load(temporary.name, map_location="cpu", weights_only=True)
    return contract, state, schedule


def _batch(dataset: Any, indices: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
    examples = [dataset[int(index)] for index in indices]
    inputs = torch.stack([example[0] for example in examples])
    targets = torch.stack([example[1] for example in examples])
    return inputs, targets


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _compare(
    generated: Mapping[str, torch.Tensor], exact: Mapping[str, torch.Tensor]
) -> dict[str, float | bool]:
    names = sorted(exact)
    error = sum(_energy(generated[name] - exact[name]) for name in names)
    exact_energy = sum(_energy(exact[name]) for name in names)
    generated_energy = sum(_energy(generated[name]) for name in names)
    dot = sum(
        float(torch.sum(generated[name].to(torch.float64) * exact[name].to(torch.float64)))
        for name in names
    )
    maximum = max(float(torch.max(torch.abs(generated[name] - exact[name]))) for name in names)
    return {
        "gradient_energy_r2": 1.0 - error / max(exact_energy, 1e-30),
        "cosine_similarity": dot / math.sqrt(max(exact_energy * generated_energy, 1e-60)),
        "relative_l2_error": math.sqrt(error / max(exact_energy, 1e-30)),
        "maximum_absolute_error": maximum,
        "bit_exact": all(torch.equal(generated[name], exact[name]) for name in names),
    }


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return type(value)((key, _to_cpu(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _keyframe_size(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
) -> int:
    payload = {
        "model": _to_cpu(model.state_dict()),
        "optimizer": _to_cpu(optimizer.state_dict()),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
    }
    with tempfile.NamedTemporaryFile(suffix=".pt") as temporary:
        torch.save(payload, temporary.name)
        temporary.flush()
        return Path(temporary.name).stat().st_size


def replay_transcript(
    transcript: Path,
    experiment: Path,
    datasets_dir: Path,
    trace: Path,
    steps: list[int],
    *,
    device_name: str = "auto",
    measure_keyframe: bool = True,
    abort_r2: float = 0.90,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    spec = _spec_from_manifest(manifest)
    contract, initial_state, schedule = read_transcript(transcript)
    device = _resolve_device(device_name)
    if str(device) != contract["runtime"]["device"]:
        raise ValueError(f"transcript requires device {contract['runtime']['device']}, got {device}")
    _seed_everything(spec.seed, bool(spec.training.get("deterministic", False)))
    task = default_task_registry().get(spec.task)
    model = task.build_model(spec)
    model.load_state_dict(initial_state)
    model = model.to(device)
    train_loader, _ = task.build_dataloaders(spec, datasets_dir)
    dataset = train_loader.dataset
    if _dataset_sha256(dataset) != contract["dataset_tokens_sha256"]:
        raise ValueError("external dataset does not match the transcript")
    optimizer = _optimizer(model, spec.optimizer)
    scheduler = _scheduler(optimizer, spec.scheduler, int(spec.training["steps"]))
    precision = str(spec.training.get("precision", "fp32")).lower()
    clip_norm = spec.training.get("gradient_clip_norm")
    exact = _read_selected_steps(trace, steps)
    recorded_losses = {
        int(row["step"]): float(row["loss"])
        for row in (
            json.loads(line)
            for line in (experiment / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        )
        if int(row["step"]) in set(steps)
    }
    requested = set(steps)
    rows: list[dict[str, Any]] = []
    keyframe_bytes: Optional[int] = None
    stopped_early = False
    started = time.perf_counter()
    model.train()
    for step in range(max(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = _move_to_device(_batch(dataset, schedule[step]), device)
        _sync(device)
        gradient_started = time.perf_counter()
        with _autocast_context(device, precision):
            loss = task.loss(model, batch, spec)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
        _sync(device)
        gradient_seconds = time.perf_counter() - gradient_started
        if step in requested:
            generated = {
                name: parameter.grad.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            metrics = _compare(generated, exact[step])
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "recorded_loss": recorded_losses[step],
                "absolute_loss_error": abs(
                    float(loss.detach().cpu()) - recorded_losses[step]
                ),
                "gradient_seconds": gradient_seconds,
                "cumulative_decode_seconds": time.perf_counter() - started,
                **metrics,
            }
            rows.append(row)
            if float(row["gradient_energy_r2"]) < abort_r2:
                stopped_early = True
                break
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if measure_keyframe and keyframe_bytes is None and step == min(steps):
            _sync(device)
            keyframe_bytes = _keyframe_size(model, optimizer, scheduler)
    total_values = sum(value.numel() for value in exact[steps[0]].values())
    training_steps = int(spec.training["steps"])
    transcript_bytes = transcript.stat().st_size
    total_raw_bytes = total_values * 4 * training_steps
    full_budget_bytes = math.floor(total_values * 0.32 / 8) * training_steps
    remaining = max(0, full_budget_bytes - transcript_bytes)
    max_extra_keyframes = 0 if not keyframe_bytes else remaining // keyframe_bytes
    passes_gate = len(rows) == len(steps) and all(
        float(row["gradient_energy_r2"]) >= 0.99 for row in rows
    )
    summary = {
        "experiment": str(experiment),
        "trace": str(trace),
        "transcript_bytes": transcript_bytes,
        "initial_model_bytes": (experiment / _INITIAL_NAME).stat().st_size,
        "training_steps": training_steps,
        "gradient_values_per_step": total_values,
        "amortized_bytes_per_step": transcript_bytes / training_steps,
        "bits_per_value": 8 * transcript_bytes / (total_values * training_steps),
        "compression_ratio_vs_fp32_gradients": total_raw_bytes / transcript_bytes,
        "hard_budget_bits_per_value": 0.32,
        "within_hard_budget": transcript_bytes <= full_budget_bytes,
        "fidelity_gate_r2": 0.99,
        "continuation_abort_r2": abort_r2,
        "minimum_sampled_r2": min(float(row["gradient_energy_r2"]) for row in rows),
        "all_sampled_steps_pass": passes_gate,
        "all_sampled_steps_bit_exact": all(bool(row["bit_exact"]) for row in rows),
        "requested_steps": steps,
        "executed_steps": [int(row["step"]) for row in rows],
        "stopped_early": stopped_early,
        "sequential_decode_seconds": time.perf_counter() - started,
        "optional_keyframe_bytes": keyframe_bytes,
        "maximum_additional_keyframes_under_budget": int(max_extra_keyframes),
        "external_side_information": contract["external_side_information"],
        "decision": (
            "procedural_side_information_control_aborted_below_continuation_cutoff"
            if stopped_early
            else "procedural_side_information_control_passes_sampled_gate"
            if passes_gate
            else "procedural_side_information_control_fails_sampled_gate"
        ),
        "rows": rows,
    }
    return rows, summary


def run_experiment(
    experiment: Path,
    datasets_dir: Path,
    steps: list[int],
    *,
    trace: Optional[Path] = None,
    transcript: Optional[Path] = None,
    device_name: str = "auto",
    measure_keyframe: bool = True,
    abort_r2: float = 0.90,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace = trace or experiment / "gradients.otsg"
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    spec = _spec_from_manifest(manifest)
    device = _resolve_device(device_name)
    _seed_everything(spec.seed, bool(spec.training.get("deterministic", False)))
    task = default_task_registry().get(spec.task)
    # Model construction is intentionally performed before loader construction,
    # matching the original runner's RNG order even though weights are reloaded.
    task.build_model(spec)
    train_loader, _ = task.build_dataloaders(spec, datasets_dir)
    schedule = sampler_schedule(
        len(train_loader.dataset),
        int(spec.training["batch_size"]),
        int(spec.training["steps"]),
        spec.seed,
    )
    if transcript is not None:
        build_transcript(
            experiment,
            transcript,
            manifest=manifest,
            dataset=train_loader.dataset,
            schedule=schedule,
            device=device,
        )
        return replay_transcript(
            transcript,
            experiment,
            datasets_dir,
            trace,
            steps,
            device_name=device_name,
            measure_keyframe=measure_keyframe,
            abort_r2=abort_r2,
        )
    with tempfile.TemporaryDirectory(prefix="ots-g1p-") as directory:
        temporary = Path(directory) / "transcript.otsg1p"
        build_transcript(
            experiment,
            temporary,
            manifest=manifest,
            dataset=train_loader.dataset,
            schedule=schedule,
            device=device,
        )
        return replay_transcript(
            temporary,
            experiment,
            datasets_dir,
            trace,
            steps,
            device_name=device_name,
            measure_keyframe=measure_keyframe,
            abort_r2=abort_r2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--steps", default="0,1,100,500,999")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--skip-keyframe-measurement", action="store_true")
    parser.add_argument("--abort-r2", type=float, default=0.90)
    args = parser.parse_args()
    rows, summary = run_experiment(
        args.experiment,
        args.datasets_dir,
        [int(value) for value in args.steps.split(",")],
        trace=args.trace,
        transcript=args.transcript,
        device_name=args.device,
        measure_keyframe=not args.skip_keyframe_measurement,
        abort_r2=args.abort_r2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
