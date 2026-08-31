"""Append-only, uncompressed storage for PyTorch gradient time series.

``.otsg`` is intentionally small and boring: a versioned JSON file header,
followed by length-delimited step records. Each record contains JSON tensor
descriptors and canonical contiguous tensor bytes. No compression is performed.
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import torch


_MAGIC = b"OTSGRAD1"
_RECORD_MAGIC = b"STEP"
_VERSION = 1
_FILE_HEADER = struct.Struct("<8sHI")
_RECORD_HEADER = struct.Struct("<4sQIQI")


class GradientTraceError(RuntimeError):
    """The trace is malformed, truncated, or violates the append contract."""


@dataclass(frozen=True)
class GradientStep:
    step: int
    gradients: Mapping[str, Optional[torch.Tensor]]


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype)
    if not name.startswith("torch."):
        raise TypeError(f"unsupported torch dtype: {dtype!r}")
    return name[len("torch.") :]


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise GradientTraceError(f"unsupported dtype in trace: {name!r}")
    return dtype


def _tensor_bytes(tensor: torch.Tensor) -> Tuple[torch.Tensor, bytes]:
    if tensor.layout != torch.strided:
        raise TypeError("only dense strided gradients are currently supported")
    value = tensor.detach().to(device="cpu").contiguous()
    raw = value.view(torch.uint8).numpy().tobytes(order="C")
    return value, raw


def _read_exact(stream: Any, size: int, context: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise GradientTraceError(f"truncated {context}")
    return value


def _read_file_header(stream: Any) -> Tuple[Dict[str, Any], bytes]:
    fixed = _read_exact(stream, _FILE_HEADER.size, "file header")
    magic, version, metadata_size = _FILE_HEADER.unpack(fixed)
    if magic != _MAGIC:
        raise GradientTraceError("not an OTS gradient trace")
    if version != _VERSION:
        raise GradientTraceError(f"unsupported gradient trace version: {version}")
    encoded_metadata = _read_exact(stream, metadata_size, "file metadata")
    try:
        metadata = json.loads(encoded_metadata)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GradientTraceError("invalid file metadata") from exc
    return metadata, fixed + encoded_metadata


class GradientTraceWriter:
    """Appends complete gradient snapshots to one trace.

    Call this immediately after ``loss.backward()`` and before gradients are
    zeroed or mutated by an optimizer. Reopening an existing trace validates all
    records and resumes after its last step.
    """

    def __init__(
        self,
        path: Path,
        *,
        experiment_id: str,
        durable: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.path = Path(path)
        self.experiment_id = experiment_id
        self.durable = durable
        extra_metadata = dict(metadata or {})
        reserved = {"experiment_id", "format"}.intersection(extra_metadata)
        if reserved:
            raise ValueError(f"reserved trace metadata keys: {sorted(reserved)}")
        self.metadata = {
            "experiment_id": experiment_id,
            "format": "ots-gradient-trace",
            **extra_metadata,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_step: Optional[int] = None

        if self.path.exists():
            self._stream = self.path.open("r+b")
            self._validate_existing()
            self._stream.seek(0, os.SEEK_END)
        else:
            self._stream = self.path.open("w+b")
            encoded_metadata = _json_bytes(self.metadata)
            self._stream.write(
                _FILE_HEADER.pack(_MAGIC, _VERSION, len(encoded_metadata))
            )
            self._stream.write(encoded_metadata)
            self._commit()

    def _validate_existing(self) -> None:
        self._stream.seek(0)
        metadata, _ = _read_file_header(self._stream)
        for key, expected in self.metadata.items():
            if metadata.get(key) != expected:
                raise GradientTraceError(
                    f"trace {key} does not match the requested value"
                )

        while True:
            fixed = self._stream.read(_RECORD_HEADER.size)
            if not fixed:
                break
            if len(fixed) != _RECORD_HEADER.size:
                raise GradientTraceError("truncated step header")
            magic, step, metadata_size, payload_size, expected_crc = (
                _RECORD_HEADER.unpack(fixed)
            )
            if magic != _RECORD_MAGIC:
                raise GradientTraceError("invalid step marker")
            metadata_bytes = _read_exact(
                self._stream, metadata_size, f"metadata for step {step}"
            )
            crc = zlib.crc32(metadata_bytes)
            remaining = payload_size
            while remaining:
                block = _read_exact(
                    self._stream,
                    min(remaining, 4 * 1024 * 1024),
                    f"payload for step {step}",
                )
                crc = zlib.crc32(block, crc)
                remaining -= len(block)
            if crc != expected_crc:
                raise GradientTraceError(f"checksum mismatch at step {step}")
            if self._last_step is not None and step <= self._last_step:
                raise GradientTraceError("steps in trace are not strictly increasing")
            self._last_step = step

    def append(
        self, step: int, gradients: Mapping[str, Optional[torch.Tensor]]
    ) -> None:
        if self._stream.closed:
            raise ValueError("cannot append to a closed gradient trace")
        if step < 0:
            raise ValueError("step must be non-negative")
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("steps must be appended in strictly increasing order")

        descriptors = []
        payload_parts = []
        offset = 0
        for name, tensor in gradients.items():
            if not name:
                raise ValueError("gradient names must be non-empty")
            if tensor is None:
                descriptors.append({"name": name, "present": False})
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"gradient {name!r} is not a torch.Tensor or None")
            value, raw = _tensor_bytes(tensor)
            descriptors.append(
                {
                    "dtype": _dtype_name(value.dtype),
                    "name": name,
                    "nbytes": len(raw),
                    "offset": offset,
                    "present": True,
                    "shape": list(value.shape),
                }
            )
            payload_parts.append(raw)
            offset += len(raw)

        metadata = _json_bytes({"tensors": descriptors})
        payload = b"".join(payload_parts)
        checksum = zlib.crc32(payload, zlib.crc32(metadata))
        header = _RECORD_HEADER.pack(
            _RECORD_MAGIC, step, len(metadata), len(payload), checksum
        )
        self._stream.write(header)
        self._stream.write(metadata)
        self._stream.write(payload)
        self._commit()
        self._last_step = step

    def _commit(self) -> None:
        self._stream.flush()
        if self.durable:
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "GradientTraceWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class GradientTraceReader:
    """Reads decoded steps or the exact raw frames consumed by compressors."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def metadata(self) -> Mapping[str, Any]:
        with self.path.open("rb") as stream:
            metadata, _ = _read_file_header(stream)
            return metadata

    def raw_header(self) -> bytes:
        with self.path.open("rb") as stream:
            _, raw = _read_file_header(stream)
            return raw

    def _raw_records(self) -> Iterator[Tuple[int, bytes, bytes, bytes]]:
        with self.path.open("rb") as stream:
            _read_file_header(stream)
            previous_step: Optional[int] = None
            while True:
                fixed = stream.read(_RECORD_HEADER.size)
                if not fixed:
                    return
                if len(fixed) != _RECORD_HEADER.size:
                    raise GradientTraceError("truncated step header")
                magic, step, metadata_size, payload_size, expected_crc = (
                    _RECORD_HEADER.unpack(fixed)
                )
                if magic != _RECORD_MAGIC:
                    raise GradientTraceError("invalid step marker")
                if previous_step is not None and step <= previous_step:
                    raise GradientTraceError("steps in trace are not strictly increasing")
                metadata = _read_exact(stream, metadata_size, f"metadata for step {step}")
                payload = _read_exact(stream, payload_size, f"payload for step {step}")
                if zlib.crc32(payload, zlib.crc32(metadata)) != expected_crc:
                    raise GradientTraceError(f"checksum mismatch at step {step}")
                previous_step = step
                yield step, fixed, metadata, payload

    def iter_raw_records(self) -> Iterator[bytes]:
        for _, fixed, metadata, payload in self._raw_records():
            yield fixed + metadata + payload

    def steps(self) -> Iterator[GradientStep]:
        for step, _, metadata_bytes, payload in self._raw_records():
            try:
                metadata = json.loads(metadata_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GradientTraceError(f"invalid metadata at step {step}") from exc

            gradients: Dict[str, Optional[torch.Tensor]] = {}
            for descriptor in metadata.get("tensors", []):
                name = descriptor["name"]
                if name in gradients:
                    raise GradientTraceError(
                        f"duplicate gradient name {name!r} at step {step}"
                    )
                if not descriptor.get("present", False):
                    gradients[name] = None
                    continue
                offset = int(descriptor["offset"])
                nbytes = int(descriptor["nbytes"])
                raw = payload[offset : offset + nbytes]
                if len(raw) != nbytes:
                    raise GradientTraceError(f"invalid tensor extent at step {step}")
                dtype = _torch_dtype(descriptor["dtype"])
                try:
                    tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                    tensor = tensor.view(dtype).reshape(descriptor["shape"]).clone()
                except (RuntimeError, ValueError) as exc:
                    raise GradientTraceError(
                        f"invalid tensor encoding for {name!r} at step {step}"
                    ) from exc
                gradients[name] = tensor
            yield GradientStep(step=step, gradients=gradients)
