"""Baseline and future research compression algorithms."""

from .baselines import (
    DeflateCompressor,
    Lz4Compressor,
    RawCompressor,
    SnappyCompressor,
    ZstdCompressor,
    available_baselines,
    baseline_registry,
)

__all__ = [
    "DeflateCompressor",
    "Lz4Compressor",
    "RawCompressor",
    "SnappyCompressor",
    "ZstdCompressor",
    "available_baselines",
    "baseline_registry",
]

