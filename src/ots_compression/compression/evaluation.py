"""Correctness-first, bounded-memory compression experiment runner."""

from __future__ import annotations

import hashlib
import json
import statistics
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import psutil

from .api import AccessMode, Compressor, PreparedState
from ..core.experiments import Experiment
from ..core.gradient_trace import GradientTraceReader


@dataclass(frozen=True)
class BenchmarkMetrics:
    algorithm: str
    configuration: Mapping[str, Any]
    access_mode: str
    uncompressed_bytes: int
    compressed_payload_bytes: int
    side_information_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    space_saving_fraction: float
    preparation_seconds: float
    encode_seconds: float
    decode_seconds: float
    encode_throughput_bytes_per_second: float
    decode_throughput_bytes_per_second: float
    mean_step_latency_seconds: Optional[float]
    p50_step_latency_seconds: Optional[float]
    p95_step_latency_seconds: Optional[float]
    average_process_rss_bytes: int
    peak_process_rss_bytes: int
    estimated_flops: Optional[int]
    exact_roundtrip: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _MemorySampler:
    def __init__(self, interval_seconds: float) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._samples: List[int] = []
        self._process = psutil.Process()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        self._samples.append(self._process.memory_info().rss)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def __enter__(self) -> "_MemorySampler":
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    @property
    def average(self) -> int:
        return round(statistics.fmean(self._samples))

    @property
    def peak(self) -> int:
        return max(self._samples)


def _file_chunks(path: Path, chunk_size: int) -> Iterator[Tuple[bytes, bool]]:
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                return
            yield chunk, False


def _online_chunks(path: Path) -> Iterator[Tuple[bytes, bool]]:
    reader = GradientTraceReader(path)
    yield reader.raw_header(), False
    for record in reader.iter_raw_records():
        yield record, True


def _percentile(values: List[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _side_information_size(state: PreparedState) -> int:
    metadata_size = 0
    if state.metadata:
        metadata_size = len(
            json.dumps(
                state.metadata, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
    return len(state.payload) + metadata_size


def benchmark_trace(
    trace_path: Path,
    compressor: Compressor,
    mode: AccessMode,
    *,
    io_chunk_size: int = 4 * 1024 * 1024,
    memory_sample_interval_seconds: float = 0.01,
) -> BenchmarkMetrics:
    """Benchmark one trace without retaining encoded or decoded data in RAM."""

    trace_path = Path(trace_path)
    if io_chunk_size <= 0:
        raise ValueError("io_chunk_size must be positive")
    if memory_sample_interval_seconds <= 0:
        raise ValueError("memory sampling interval must be positive")

    completed_trace = trace_path if mode is AccessMode.OFFLINE else None
    preparation_started = time.perf_counter()
    state = compressor.prepare(mode, completed_trace)
    preparation_seconds = time.perf_counter() - preparation_started
    side_information_bytes = _side_information_size(state)

    source_chunks: Iterable[Tuple[bytes, bool]]
    if mode is AccessMode.ONLINE:
        source_chunks = _online_chunks(trace_path)
    else:
        source_chunks = _file_chunks(trace_path, io_chunk_size)

    source_hash = hashlib.sha256()
    uncompressed_bytes = 0
    step_latencies: List[float] = []
    encoder = compressor.encoder(mode, state)

    with tempfile.TemporaryFile(mode="w+b") as encoded_stream:
        with _MemorySampler(memory_sample_interval_seconds) as memory:
            encode_started = time.perf_counter()
            for chunk, checkpoint in source_chunks:
                source_hash.update(chunk)
                uncompressed_bytes += len(chunk)
                push_started = time.perf_counter()
                encoded = encoder.push(chunk, checkpoint=checkpoint)
                push_seconds = time.perf_counter() - push_started
                encoded_stream.write(encoded)
                if checkpoint:
                    step_latencies.append(push_seconds)
            encoded_stream.write(encoder.finish())
            encode_seconds = time.perf_counter() - encode_started
        compressed_payload_bytes = encoded_stream.tell()

        decoded_hash = hashlib.sha256()
        decoded_bytes = 0
        decoder = compressor.decoder(mode, state)
        encoded_stream.seek(0)
        decode_started = time.perf_counter()
        while True:
            encoded = encoded_stream.read(io_chunk_size)
            if not encoded:
                break
            decoded = decoder.push(encoded)
            decoded_hash.update(decoded)
            decoded_bytes += len(decoded)
        decoded = decoder.finish()
        decoded_hash.update(decoded)
        decoded_bytes += len(decoded)
        decode_seconds = time.perf_counter() - decode_started

    exact_roundtrip = (
        decoded_bytes == uncompressed_bytes
        and decoded_hash.digest() == source_hash.digest()
    )
    if not exact_roundtrip:
        raise ValueError(f"{compressor.name} failed exact round-trip verification")

    total_compressed_bytes = compressed_payload_bytes + side_information_bytes
    ratio = (
        uncompressed_bytes / total_compressed_bytes
        if total_compressed_bytes
        else float("inf")
    )
    saving = (
        1.0 - total_compressed_bytes / uncompressed_bytes
        if uncompressed_bytes
        else 0.0
    )

    return BenchmarkMetrics(
        algorithm=compressor.name,
        configuration=compressor.configuration(),
        access_mode=mode.value,
        uncompressed_bytes=uncompressed_bytes,
        compressed_payload_bytes=compressed_payload_bytes,
        side_information_bytes=side_information_bytes,
        total_compressed_bytes=total_compressed_bytes,
        compression_ratio=ratio,
        space_saving_fraction=saving,
        preparation_seconds=preparation_seconds,
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        encode_throughput_bytes_per_second=(
            uncompressed_bytes / encode_seconds if encode_seconds else float("inf")
        ),
        decode_throughput_bytes_per_second=(
            uncompressed_bytes / decode_seconds if decode_seconds else float("inf")
        ),
        mean_step_latency_seconds=(
            statistics.fmean(step_latencies) if step_latencies else None
        ),
        p50_step_latency_seconds=_percentile(step_latencies, 0.50),
        p95_step_latency_seconds=_percentile(step_latencies, 0.95),
        average_process_rss_bytes=memory.average,
        peak_process_rss_bytes=memory.peak,
        estimated_flops=compressor.estimated_flops(uncompressed_bytes),
        exact_roundtrip=True,
    )


def write_metrics(metrics: BenchmarkMetrics, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def benchmark_experiment(
    experiment: Experiment,
    compressor: Compressor,
    mode: AccessMode,
) -> BenchmarkMetrics:
    metrics = benchmark_trace(experiment.gradients_path, compressor, mode)
    destination = (
        experiment.baseline_results_path
        / compressor.name
        / mode.value
        / "metrics.json"
    )
    write_metrics(metrics, destination)
    return metrics
