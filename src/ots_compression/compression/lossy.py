"""Lossy trace experiment helpers with explicit gradient-fidelity metrics."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from ..core.gradient_trace import GradientTraceReader
from .algorithms.ots_deltaq import OTSDeltaQCompressor


@dataclass(frozen=True)
class LossyMetrics:
    algorithm: str
    configuration: dict
    source_bytes: int
    compressed_bytes: int
    compression_ratio: float
    encode_seconds: float
    decode_seconds: float
    gradient_energy_r2: float
    gradient_cosine_similarity: float
    relative_squared_error: float
    decoded_trace: str
    exact_roundtrip: bool = False
    codec_stats: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


def benchmark_deltaq(trace_path: Path, compressor: OTSDeltaQCompressor, destination: Path) -> LossyMetrics:
    """Encode, decode, and measure a lossy artifact without claiming exactness."""

    trace_path, destination = Path(trace_path), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / "compressed.otsdq"
    decoded = destination / "decoded_gradients.otsg"
    started = time.perf_counter()
    codec_stats = compressor.compress(
        trace_path,
        artifact,
        predictions_path=destination / "predictions.otsg",
        prediction_metrics_path=destination / "prediction_metrics.jsonl",
    )
    encode_seconds = time.perf_counter() - started
    return evaluate_deltaq_artifact(
        trace_path,
        compressor,
        artifact,
        destination,
        encode_seconds=encode_seconds,
        codec_stats=codec_stats.to_dict(),
    )


def evaluate_deltaq_artifact(
    trace_path: Path,
    compressor: OTSDeltaQCompressor,
    artifact: Path,
    destination: Path,
    *,
    encode_seconds: float,
    codec_stats: Optional[dict] = None,
) -> LossyMetrics:
    """Decode and score an artifact that was produced during training."""

    trace_path, artifact, destination = (
        Path(trace_path),
        Path(artifact),
        Path(destination),
    )
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    decoded = destination / "decoded_gradients.otsg"
    compressor.decompress(artifact, decoded)
    decode_seconds = time.perf_counter() - started
    error, energy, dot, decoded_energy = _compare_traces(trace_path, decoded)
    metrics = LossyMetrics(
        algorithm=compressor.name,
        configuration=compressor.configuration(),
        source_bytes=trace_path.stat().st_size,
        compressed_bytes=artifact.stat().st_size,
        compression_ratio=trace_path.stat().st_size / artifact.stat().st_size,
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        gradient_energy_r2=1.0 - error / max(energy, 1e-30),
        gradient_cosine_similarity=dot / max((energy * decoded_energy) ** 0.5, 1e-30),
        relative_squared_error=error / max(energy, 1e-30),
        decoded_trace=str(decoded),
        codec_stats=codec_stats,
    )
    (destination / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _compare_traces(source: Path, decoded: Path) -> tuple[float, float, float, float]:
    source_steps, decoded_steps = GradientTraceReader(source).steps(), GradientTraceReader(decoded).steps()
    error = energy = dot = decoded_energy = 0.0
    for original, restored in zip(source_steps, decoded_steps):
        if original.step != restored.step or original.gradients.keys() != restored.gradients.keys():
            raise ValueError("decoded trace schema differs from source")
        for name, value in original.gradients.items():
            recovered = restored.gradients[name]
            if value is None and recovered is None:
                continue
            if value is None or recovered is None or value.shape != recovered.shape:
                raise ValueError(f"decoded gradient mismatch for {name}")
            a, b = value.to(torch.float64), recovered.to(torch.float64)
            error += float(torch.sum((a - b).square()))
            energy += float(torch.sum(a.square()))
            dot += float(torch.sum(a * b))
            decoded_energy += float(torch.sum(b.square()))
    if next(source_steps, None) is not None or next(decoded_steps, None) is not None:
        raise ValueError("decoded trace has a different step count")
    return error, energy, dot, decoded_energy


def refresh_codec_stats(metrics_path: Path, codec_stats: dict) -> None:
    """Attach corrected artifact-inspection statistics without rerunning it."""

    metrics_path = Path(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["codec_stats"] = codec_stats
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def compare_prediction_traces(
    source: Path, predictions: Path, output: Path
) -> None:
    """Write plot-ready prediction differences for every recorded step."""

    source_steps = GradientTraceReader(Path(source)).steps()
    prediction_steps = GradientTraceReader(Path(predictions)).steps()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w", encoding="utf-8") as stream:
        for original, predicted in zip(source_steps, prediction_steps):
            if (
                original.step != predicted.step
                or original.gradients.keys() != predicted.gradients.keys()
            ):
                raise ValueError("prediction trace schema differs from source")
            error = energy = prediction_energy = dot = 0.0
            elements = 0
            for name, value in original.gradients.items():
                estimate = predicted.gradients[name]
                if value is None and estimate is None:
                    continue
                if value is None or estimate is None or value.shape != estimate.shape:
                    raise ValueError(f"prediction mismatch for {name}")
                actual = value.to(torch.float64)
                guess = estimate.to(torch.float64)
                error += float(torch.sum((actual - guess).square()))
                energy += float(torch.sum(actual.square()))
                prediction_energy += float(torch.sum(guess.square()))
                dot += float(torch.sum(actual * guess))
                elements += actual.numel()
            row = {
                "step": original.step,
                "elements": elements,
                "prediction_energy_r2": 1.0 - error / max(energy, 1e-30),
                "prediction_cosine_similarity": dot
                / max((energy * prediction_energy) ** 0.5, 1e-30),
                "prediction_relative_l2": (error / max(energy, 1e-30)) ** 0.5,
            }
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    if next(source_steps, None) is not None or next(prediction_steps, None) is not None:
        raise ValueError("prediction trace has a different step count")
