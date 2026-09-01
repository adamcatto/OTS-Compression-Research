"""Exact D4 adaptive-predictor codec construction."""

from ots_compression.compression.algorithms.ots_deltaq import OTSDeltaQCompressor


def build_algorithm() -> OTSDeltaQCompressor:
    return OTSDeltaQCompressor(
        block_size=16_384,
        relative_squared_error=1e-4,
        prediction="adaptive_zero_previous",
    )
