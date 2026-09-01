"""Exact E1 fixed-initializer rank-1 codec construction."""

from ots_compression.compression.algorithms.ots_deltaq import OTSDeltaQCompressor


def build_algorithm() -> OTSDeltaQCompressor:
    return OTSDeltaQCompressor(
        block_size=16_384,
        relative_squared_error=1e-4,
        prediction="rank1_tensor",
        rank1_power_iterations=1,
        rank1_warm_start=False,
    )
