"""Lossless control constructions used by L0."""

from ots_compression.compression.algorithms.baselines import (
    DeflateCompressor,
    Lz4Compressor,
    RawCompressor,
    SnappyCompressor,
    ZstdCompressor,
)


def algorithm_factories():
    # Factories defer importing optional codec runtimes until selected.
    return {
        "raw": RawCompressor,
        "deflate": lambda: DeflateCompressor(level=6),
        "zstd": lambda: ZstdCompressor(level=3),
        "lz4": lambda: Lz4Compressor(compression_level=0),
        "snappy": SnappyCompressor,
    }
