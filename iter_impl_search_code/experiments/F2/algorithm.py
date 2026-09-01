"""Exact compressor construction for F2 on experiment_003."""

from ots_compression.compression.algorithms.ots_deltaq import OTSDeltaQCompressor


def build_compressor() -> OTSDeltaQCompressor:
    return OTSDeltaQCompressor(
        block_size=16_384,
        relative_squared_error=1e-4,
        prediction="zero",
        block_transform="randomized_hadamard",
        allocation="optimizer_aware_global",
        preconditioned_relative_squared_error=1e-4,
        sensitivity_beta2=0.95,
        sensitivity_epsilon=1e-8,
        allocator_iterations=40,
    )
