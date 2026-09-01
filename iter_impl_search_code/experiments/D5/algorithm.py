"""Exact D5 sparse-outlier codec construction."""

from ots_compression.compression.algorithms.ots_deltaq import OTSDeltaQCompressor


def build_algorithm() -> OTSDeltaQCompressor:
    return OTSDeltaQCompressor(
        block_size=16_384,
        relative_squared_error=1e-4,
        prediction="previous_decoded_gradient",
        outlier_fraction=0.01,
    )
