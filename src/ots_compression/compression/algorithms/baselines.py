"""Adapters for established general-purpose lossless compressors."""

from __future__ import annotations

import zlib
from typing import Any, Dict, Mapping, Tuple

from ..api import (
    AccessMode,
    Compressor,
    Decoder,
    Encoder,
    MissingCodecDependency,
    PreparedState,
)


class _RawEncoder(Encoder):
    def push(self, data: bytes, *, checkpoint: bool) -> bytes:
        return data

    def finish(self) -> bytes:
        return b""


class _RawDecoder(Decoder):
    def push(self, data: bytes) -> bytes:
        return data

    def finish(self) -> bytes:
        return b""


class RawCompressor(Compressor):
    """No-compression control for framing and I/O overhead."""

    name = "raw"

    def encoder(self, mode: AccessMode, state: PreparedState) -> Encoder:
        return _RawEncoder()

    def decoder(self, mode: AccessMode, state: PreparedState) -> Decoder:
        return _RawDecoder()


class _DeflateEncoder(Encoder):
    def __init__(self, level: int) -> None:
        self._encoder = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
        self._finished = False

    def push(self, data: bytes, *, checkpoint: bool) -> bytes:
        if self._finished:
            raise ValueError("encoder is already finished")
        encoded = self._encoder.compress(data)
        if checkpoint:
            encoded += self._encoder.flush(zlib.Z_SYNC_FLUSH)
        return encoded

    def finish(self) -> bytes:
        if self._finished:
            return b""
        self._finished = True
        return self._encoder.flush(zlib.Z_FINISH)


class _DeflateDecoder(Decoder):
    def __init__(self) -> None:
        self._decoder = zlib.decompressobj(-zlib.MAX_WBITS)

    def push(self, data: bytes) -> bytes:
        return self._decoder.decompress(data)

    def finish(self) -> bytes:
        decoded = self._decoder.flush()
        if not self._decoder.eof:
            raise ValueError("incomplete DEFLATE stream")
        return decoded


class DeflateCompressor(Compressor):
    name = "deflate"

    def __init__(self, level: int = 6) -> None:
        if not -1 <= level <= 9:
            raise ValueError("DEFLATE level must be between -1 and 9")
        self.level = level

    def configuration(self) -> Mapping[str, Any]:
        return {"level": self.level}

    def encoder(self, mode: AccessMode, state: PreparedState) -> Encoder:
        return _DeflateEncoder(self.level)

    def decoder(self, mode: AccessMode, state: PreparedState) -> Decoder:
        return _DeflateDecoder()


class _ZstdEncoder(Encoder):
    def __init__(self, module: Any, level: int) -> None:
        self._module = module
        self._encoder = module.ZstdCompressor(level=level).compressobj()
        self._finished = False

    def push(self, data: bytes, *, checkpoint: bool) -> bytes:
        if self._finished:
            raise ValueError("encoder is already finished")
        encoded = self._encoder.compress(data)
        if checkpoint:
            encoded += self._encoder.flush(self._module.COMPRESSOBJ_FLUSH_BLOCK)
        return encoded

    def finish(self) -> bytes:
        if self._finished:
            return b""
        self._finished = True
        return self._encoder.flush(self._module.COMPRESSOBJ_FLUSH_FINISH)


class _ZstdDecoder(Decoder):
    def __init__(self, module: Any) -> None:
        self._decoder = module.ZstdDecompressor().decompressobj()

    def push(self, data: bytes) -> bytes:
        return self._decoder.decompress(data)

    def finish(self) -> bytes:
        decoded = self._decoder.flush()
        if not self._decoder.eof:
            raise ValueError("incomplete Zstandard stream")
        return decoded


class ZstdCompressor(Compressor):
    name = "zstd"

    def __init__(self, level: int = 3) -> None:
        self.level = level
        try:
            import zstandard
        except ImportError as exc:
            raise MissingCodecDependency(
                "Zstandard requires the 'zstandard' package"
            ) from exc
        self._module = zstandard

    def configuration(self) -> Mapping[str, Any]:
        return {"level": self.level}

    def encoder(self, mode: AccessMode, state: PreparedState) -> Encoder:
        return _ZstdEncoder(self._module, self.level)

    def decoder(self, mode: AccessMode, state: PreparedState) -> Decoder:
        return _ZstdDecoder(self._module)


class _Lz4Encoder(Encoder):
    def __init__(self, module: Any, online: bool, compression_level: int) -> None:
        self._encoder = module.LZ4FrameCompressor(
            compression_level=compression_level,
            auto_flush=online,
        )
        self._prefix = self._encoder.begin()
        self._finished = False

    def push(self, data: bytes, *, checkpoint: bool) -> bytes:
        if self._finished:
            raise ValueError("encoder is already finished")
        prefix, self._prefix = self._prefix, b""
        return prefix + self._encoder.compress(data)

    def finish(self) -> bytes:
        if self._finished:
            return b""
        self._finished = True
        prefix, self._prefix = self._prefix, b""
        return prefix + self._encoder.flush()


class _Lz4Decoder(Decoder):
    def __init__(self, module: Any) -> None:
        self._decoder = module.LZ4FrameDecompressor()

    def push(self, data: bytes) -> bytes:
        return self._decoder.decompress(data)

    def finish(self) -> bytes:
        if not self._decoder.eof:
            raise ValueError("incomplete LZ4 frame")
        return b""


class Lz4Compressor(Compressor):
    name = "lz4"

    def __init__(self, compression_level: int = 0) -> None:
        self.compression_level = compression_level
        try:
            import lz4.frame
        except ImportError as exc:
            raise MissingCodecDependency("LZ4 requires the 'lz4' package") from exc
        self._module = lz4.frame

    def configuration(self) -> Mapping[str, Any]:
        return {"compression_level": self.compression_level}

    def encoder(self, mode: AccessMode, state: PreparedState) -> Encoder:
        return _Lz4Encoder(
            self._module,
            online=mode is AccessMode.ONLINE,
            compression_level=self.compression_level,
        )

    def decoder(self, mode: AccessMode, state: PreparedState) -> Decoder:
        return _Lz4Decoder(self._module)


class _SnappyEncoder(Encoder):
    def __init__(self, module: Any) -> None:
        self._encoder = module.StreamCompressor()

    def push(self, data: bytes, *, checkpoint: bool) -> bytes:
        return self._encoder.add_chunk(data, compress=True)

    def finish(self) -> bytes:
        return b""


class _SnappyDecoder(Decoder):
    def __init__(self, module: Any) -> None:
        self._decoder = module.StreamDecompressor()

    def push(self, data: bytes) -> bytes:
        return self._decoder.decompress(data)

    def finish(self) -> bytes:
        return b""


class SnappyCompressor(Compressor):
    name = "snappy"

    def __init__(self) -> None:
        try:
            import snappy
        except ImportError as exc:
            raise MissingCodecDependency(
                "Snappy requires the 'python-snappy' package"
            ) from exc
        self._module = snappy

    def encoder(self, mode: AccessMode, state: PreparedState) -> Encoder:
        return _SnappyEncoder(self._module)

    def decoder(self, mode: AccessMode, state: PreparedState) -> Decoder:
        return _SnappyDecoder(self._module)


def available_baselines() -> Tuple[Compressor, ...]:
    """Instantiate every baseline available in the current environment."""

    compressors = [RawCompressor(), DeflateCompressor()]
    for compressor_type in (ZstdCompressor, Lz4Compressor, SnappyCompressor):
        try:
            compressors.append(compressor_type())
        except MissingCodecDependency:
            pass
    return tuple(compressors)


def baseline_registry() -> Dict[str, Compressor]:
    return {compressor.name: compressor for compressor in available_baselines()}
