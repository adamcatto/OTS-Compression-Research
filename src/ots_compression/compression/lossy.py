"""Lossy trace experiment helpers with explicit gradient-fidelity metrics."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

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

    def to_dict(self) -> dict:
        return asdict(self)


def benchmark_deltaq(trace_path: Path, compressor: OTSDeltaQCompressor, destination: Path) -> LossyMetrics:
    """Encode, decode, and measure a lossy artifact without claiming exactness."""

    trace_path, destination = Path(trace_path), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / "compressed.otsdq"
    decoded = destination / "decoded_gradients.otsg"
    started = time.perf_counter()
    compressor.compress(trace_path, artifact)
    encode_seconds = time.perf_counter() - started
    return evaluate_deltaq_artifact(
        trace_path,
        compressor,
        artifact,
        destination,
        encode_seconds=encode_seconds,
    )


def evaluate_deltaq_artifact(
    trace_path: Path,
    compressor: OTSDeltaQCompressor,
    artifact: Path,
    destination: Path,
    *,
    encode_seconds: float,
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
