"""Common streaming contract for baseline, learned, and searched compressors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional


class AccessMode(str, Enum):
    """Information available before and during compression."""

    ONLINE = "online"
    OFFLINE = "offline"


@dataclass(frozen=True)
class PreparedState:
    """Dictionary/model/side information required for decoding.

    ``payload`` is counted in compressed size. Metadata is descriptive and is
    also serialized by persistent result writers.
    """

    payload: bytes = b""
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Encoder(ABC):
    """Consumes source bytes in order and returns bytes ready to append."""

    @abstractmethod
    def push(self, data: bytes, *, checkpoint: bool) -> bytes:
        """Encode bytes; at a checkpoint, commit output without future input."""

    @abstractmethod
    def finish(self) -> bytes:
        """Finalize the encoded stream."""


class Decoder(ABC):
    @abstractmethod
    def push(self, data: bytes) -> bytes:
        """Decode the next arbitrary segment of the encoded stream."""

    @abstractmethod
    def finish(self) -> bytes:
        """Finalize decoding and validate the stream where supported."""


class Compressor(ABC):
    """Compression algorithm boundary used by every benchmark candidate.

    For offline evaluation, ``prepare`` receives the completed trace path and
    may inspect it multiple times. For online evaluation it receives ``None``;
    the encoder then sees only the header and each newly appended step.
    """

    name: str

    def configuration(self) -> Mapping[str, Any]:
        return {}

    def prepare(
        self, mode: AccessMode, trace_path: Optional[Path]
    ) -> PreparedState:
        if mode is AccessMode.ONLINE and trace_path is not None:
            raise ValueError("online preparation cannot inspect a completed trace")
        if mode is AccessMode.OFFLINE and trace_path is None:
            raise ValueError("offline preparation requires the completed trace")
        return PreparedState()

    @abstractmethod
    def encoder(self, mode: AccessMode, state: PreparedState) -> Encoder:
        raise NotImplementedError

    @abstractmethod
    def decoder(self, mode: AccessMode, state: PreparedState) -> Decoder:
        raise NotImplementedError

    def estimated_flops(self, uncompressed_bytes: int) -> Optional[int]:
        """Return an analytic estimate, or ``None`` when it is not meaningful."""

        return None


class MissingCodecDependency(RuntimeError):
    pass
