"""Exact E2 algorithm construction used by experiment_003."""

from __future__ import annotations

import json

from ots_compression.compression.algorithms.ots_deltaq import OTSDeltaQCompressor


def build_algorithm() -> OTSDeltaQCompressor:
    return OTSDeltaQCompressor(
        block_size=16_384,
        relative_squared_error=1e-4,
        prediction="rank1_tensor",
        rank1_power_iterations=1,
        rank1_warm_start=True,
    )


if __name__ == "__main__":
    algorithm = build_algorithm()
    print(json.dumps({"name": algorithm.name, **algorithm.configuration()}, indent=2))
