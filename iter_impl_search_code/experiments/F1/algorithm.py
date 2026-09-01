"""Exact compressor construction for F1."""

from ots_compression.compression.algorithms.ots_deltaq import OTSDeltaQCompressor


def build_compressor() -> OTSDeltaQCompressor:
    return OTSDeltaQCompressor(
        block_size=16_384,
        relative_squared_error=1e-4,
        prediction="zero",
        block_transform="randomized_hadamard",
    )
