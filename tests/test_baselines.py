from pathlib import Path

import pytest
import torch

from ots_compression.compression import AccessMode, MissingCodecDependency
from ots_compression.compression.algorithms.baselines import (
    DeflateCompressor,
    RawCompressor,
    ZstdCompressor,
)
from ots_compression.compression.evaluation import benchmark_trace
from ots_compression.core.gradient_trace import GradientTraceWriter


@pytest.fixture
def trace_path(tmp_path: Path) -> Path:
    path = tmp_path / "gradients.otsg"
    with GradientTraceWriter(path, experiment_id="experiment_001") as writer:
        for step in range(4):
            writer.append(
                step,
                {"weight": torch.arange(1024, dtype=torch.float32) + step},
            )
    return path


@pytest.mark.parametrize("mode", list(AccessMode))
@pytest.mark.parametrize("compressor", [RawCompressor(), DeflateCompressor()])
def test_baselines_round_trip(trace_path: Path, mode: AccessMode, compressor) -> None:
    metrics = benchmark_trace(trace_path, compressor, mode, io_chunk_size=128)

    assert metrics.exact_roundtrip
    assert metrics.uncompressed_bytes == trace_path.stat().st_size
    assert metrics.total_compressed_bytes > 0
    if mode is AccessMode.ONLINE:
        assert metrics.mean_step_latency_seconds is not None
    else:
        assert metrics.mean_step_latency_seconds is None


@pytest.mark.parametrize("mode", list(AccessMode))
def test_zstd_round_trip_when_available(trace_path: Path, mode: AccessMode) -> None:
    try:
        compressor = ZstdCompressor()
    except MissingCodecDependency:
        pytest.skip("zstandard is not installed")
    metrics = benchmark_trace(trace_path, compressor, mode)
    assert metrics.exact_roundtrip
