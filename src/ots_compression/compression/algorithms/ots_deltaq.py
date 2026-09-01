"""OTS-DeltaQ, an online decoder-consistent lossy gradient codec.

The first implementation prioritizes a falsifiable fidelity guarantee over an
aggressive learned model: each tensor is predicted from its previously decoded
value, then each residual block uses int8 only when its reconstruction error
fits a configured relative squared-error budget.  Otherwise it falls back to
float16 or float32.  The decoder updates exactly the same predictor.
"""

from __future__ import annotations

import io
import json
import math
import statistics
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ...core.gradient_trace import GradientStep, GradientTraceReader, GradientTraceWriter


_MAGIC = b"OTSDQ001"
_VERSION = 5
_FILE_HEADER = struct.Struct("<8sHI")
_RECORD_HEADER = struct.Struct("<4sQIQ")
_RECORD_MAGIC = b"STEP"
_BLOCK_HEADER = struct.Struct("<Bf")
_INT8, _FLOAT16, _FLOAT32 = range(3)
_INT8_OUTLIER = 3
_OUTLIER_COUNT = struct.Struct("<I")
_RHT_FLAG = 0x08


@dataclass(frozen=True)
class DeltaQStats:
    steps: int
    tensors: int
    int8_blocks: int
    float16_blocks: int
    float32_blocks: int
    int8_elements: int
    float16_elements: int
    float32_elements: int
    previous_prediction_blocks: int
    zero_prediction_blocks: int
    previous_prediction_elements: int
    zero_prediction_elements: int
    int8_outlier_blocks: int
    int8_outlier_elements: int
    outlier_values: int
    rank1_prediction_tensors: int
    rank1_prediction_blocks: int
    rank1_prediction_elements: int
    rank1_factor_bytes: int
    rht_int8_blocks: int
    rht_int8_elements: int

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "tensors": self.tensors,
            "int8_blocks": self.int8_blocks,
            "float16_blocks": self.float16_blocks,
            "float32_blocks": self.float32_blocks,
            "int8_elements": self.int8_elements,
            "float16_elements": self.float16_elements,
            "float32_elements": self.float32_elements,
            "previous_prediction_blocks": self.previous_prediction_blocks,
            "zero_prediction_blocks": self.zero_prediction_blocks,
            "previous_prediction_elements": self.previous_prediction_elements,
            "zero_prediction_elements": self.zero_prediction_elements,
            "int8_outlier_blocks": self.int8_outlier_blocks,
            "int8_outlier_elements": self.int8_outlier_elements,
            "outlier_values": self.outlier_values,
            "rank1_prediction_tensors": self.rank1_prediction_tensors,
            "rank1_prediction_blocks": self.rank1_prediction_blocks,
            "rank1_prediction_elements": self.rank1_prediction_elements,
            "rank1_factor_bytes": self.rank1_factor_bytes,
            "rht_int8_blocks": self.rht_int8_blocks,
            "rht_int8_elements": self.rht_int8_elements,
        }


@dataclass(frozen=True)
class _F2Candidate:
    """One independently decodable rate-distortion point for a block."""

    mode: int
    scale: float
    encoded: bytes
    restored: torch.Tensor
    squared_error: float
    transformed: bool

    @property
    def payload_nbytes(self) -> int:
        return _BLOCK_HEADER.size + len(self.encoded)


@dataclass(frozen=True)
class _F2Block:
    """Block frontier plus the state-derived optimizer sensitivity proxy."""

    tensor_name: str
    offset: int
    count: int
    energy: float
    sensitivity_weight: float
    candidates: Tuple[_F2Candidate, ...]


class OTSDeltaQOnlineWriter:
    """Append one encoded record immediately after each captured gradient step."""

    def __init__(
        self,
        compressor: "OTSDeltaQCompressor",
        artifact_path: Path,
        *,
        source_metadata: Mapping[str, Any],
        predictions_path: Optional[Path] = None,
        prediction_metrics_path: Optional[Path] = None,
        summary_path: Optional[Path] = None,
        durable: bool = False,
        access_mode: str = "online",
    ) -> None:
        self.compressor = compressor
        self.artifact_path = Path(artifact_path)
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.artifact_path.with_suffix(
            self.artifact_path.suffix + ".tmp"
        )
        self._stream = self.temporary_path.open("wb")
        self._durable = durable
        self._predictors: Dict[str, torch.Tensor] = {}
        self._rank1_vectors: Dict[str, torch.Tensor] = {}
        self._sensitivity_state: Dict[str, torch.Tensor] = {}
        self._last_step: Optional[int] = None
        self._started = time.perf_counter()
        self._counters = {
            "steps": 0,
            "tensors": 0,
            "int8_blocks": 0,
            "float16_blocks": 0,
            "float32_blocks": 0,
            "int8_elements": 0,
            "float16_elements": 0,
            "float32_elements": 0,
            "previous_prediction_blocks": 0,
            "zero_prediction_blocks": 0,
            "previous_prediction_elements": 0,
            "zero_prediction_elements": 0,
            "int8_outlier_blocks": 0,
            "int8_outlier_elements": 0,
            "outlier_values": 0,
            "rank1_prediction_tensors": 0,
            "rank1_prediction_blocks": 0,
            "rank1_prediction_elements": 0,
            "rank1_factor_bytes": 0,
            "rht_int8_blocks": 0,
            "rht_int8_elements": 0,
        }
        self._summary_path = Path(summary_path) if summary_path else None
        self._access_mode = access_mode
        self._prediction_metrics_path = (
            Path(prediction_metrics_path) if prediction_metrics_path else None
        )
        self._prediction_metrics_stream = None
        if self._prediction_metrics_path is not None:
            self._prediction_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._prediction_metrics_stream = self._prediction_metrics_path.open(
                "w", encoding="utf-8"
            )
        self._prediction_writer = None
        if predictions_path is not None:
            extra_metadata = {
                "gradient_capture": source_metadata.get("gradient_capture"),
                "kind": "deltaq_decoder_predictions",
                "algorithm": compressor.name,
            }
            self._prediction_writer = GradientTraceWriter(
                Path(predictions_path),
                experiment_id=str(source_metadata["experiment_id"]),
                durable=durable,
                metadata={key: value for key, value in extra_metadata.items() if value is not None},
            )
        header = {
            "format": "ots-deltaq",
            "version": _VERSION,
            "access_mode": access_mode,
            "source_metadata": dict(source_metadata),
            "configuration": compressor.configuration(),
        }
        encoded_header = _canonical_json(header)
        self._stream.write(_FILE_HEADER.pack(_MAGIC, _VERSION, len(encoded_header)))
        self._stream.write(encoded_header)
        self._commit()

    def append(
        self, step: int, gradients: Mapping[str, Optional[torch.Tensor]]
    ) -> None:
        if self._stream.closed:
            raise ValueError("cannot append to a closed OTS-DeltaQ writer")
        if step < 0 or (self._last_step is not None and step <= self._last_step):
            raise ValueError("steps must be nonnegative and strictly increasing")
        started = time.perf_counter()
        before = dict(self._counters)
        metadata, payload, decoded, predictions = self.compressor._encode_step(
            GradientStep(step=step, gradients=gradients),
            self._predictors,
            self._counters,
            self._rank1_vectors,
            self._sensitivity_state,
        )
        fixed = _RECORD_HEADER.pack(
            _RECORD_MAGIC, step, len(metadata), len(payload)
        )
        self._stream.write(fixed)
        self._stream.write(metadata)
        self._stream.write(payload)
        self._commit()
        codec_seconds = time.perf_counter() - started
        if self._prediction_writer is not None:
            self._prediction_writer.append(step, predictions)
        if self._prediction_metrics_stream is not None:
            metric = _step_metrics(
                step,
                gradients,
                predictions,
                decoded,
                encoded_bytes=len(fixed) + len(metadata) + len(payload),
                block_counts={
                    key: self._counters[key] - before[key]
                    for key in (
                        "int8_blocks",
                        "float16_blocks",
                        "float32_blocks",
                        "int8_elements",
                        "float16_elements",
                        "float32_elements",
                        "previous_prediction_blocks",
                        "zero_prediction_blocks",
                        "previous_prediction_elements",
                        "zero_prediction_elements",
                        "int8_outlier_blocks",
                        "int8_outlier_elements",
                        "outlier_values",
                        "rank1_prediction_tensors",
                        "rank1_prediction_blocks",
                        "rank1_prediction_elements",
                        "rank1_factor_bytes",
                        "rht_int8_blocks",
                        "rht_int8_elements",
                    )
                },
                encode_seconds=codec_seconds,
                extra_metrics=self.compressor._last_allocator_metrics,
            )
            self._prediction_metrics_stream.write(
                json.dumps(metric, sort_keys=True) + "\n"
            )
            self._prediction_metrics_stream.flush()
        self._predictors.update(decoded)
        self._counters["steps"] += 1
        self._last_step = step

    def _commit(self) -> None:
        self._stream.flush()
        if self._durable:
            import os

            os.fsync(self._stream.fileno())

    def close(self, *, commit: bool = True) -> DeltaQStats:
        if self._stream.closed:
            return DeltaQStats(**self._counters)
        self._stream.close()
        if self._prediction_writer is not None:
            self._prediction_writer.close()
        if self._prediction_metrics_stream is not None:
            self._prediction_metrics_stream.close()
        if commit:
            self.temporary_path.replace(self.artifact_path)
            if self._summary_path is not None:
                summary = {
                    "access_mode": self._access_mode,
                    "algorithm": self.compressor.name,
                    "artifact": str(self.artifact_path),
                    "compressed_bytes": self.artifact_path.stat().st_size,
                    "configuration": self.compressor.configuration(),
                    "encode_wall_seconds": time.perf_counter() - self._started,
                    **self._counters,
                }
                if self._prediction_metrics_path is not None:
                    summary["prediction_metrics"] = summarize_prediction_metrics(
                        self._prediction_metrics_path
                    )
                self._summary_path.parent.mkdir(parents=True, exist_ok=True)
                self._summary_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        return DeltaQStats(**self._counters)

    def __enter__(self) -> "OTSDeltaQOnlineWriter":
        return self

    def __exit__(self, exc_type, *_: Any) -> None:
        self.close(commit=exc_type is None)


class OTSDeltaQCompressor:
    """Blockwise temporal residual codec suitable for sharded online use.

    The legacy ``per_block`` allocator checks ``relative_squared_error`` for
    every block independently.  F2's ``optimizer_aware_global`` allocator
    instead enforces raw and sensitivity-weighted bounds across each emitted
    step while choosing block modes jointly.
    """

    def __init__(
        self,
        *,
        block_size: int = 16_384,
        relative_squared_error: float = 1e-4,
        prediction: str = "previous_decoded_gradient",
        outlier_fraction: float = 0.0,
        rank1_power_iterations: int = 1,
        rank1_warm_start: bool = False,
        block_transform: str = "none",
        allocation: str = "per_block",
        preconditioned_relative_squared_error: float = 1e-4,
        sensitivity_beta2: float = 0.999,
        sensitivity_epsilon: float = 1e-8,
        allocator_iterations: int = 40,
    ):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0 < relative_squared_error < 1:
            raise ValueError("relative_squared_error must be in (0, 1)")
        if prediction not in {
            "previous_decoded_gradient",
            "zero",
            "adaptive_zero_previous",
            "rank1_tensor",
        }:
            raise ValueError(
                "prediction must be 'previous_decoded_gradient', 'zero', "
                "'adaptive_zero_previous', or 'rank1_tensor'"
            )
        if not 0.0 <= outlier_fraction < 1.0:
            raise ValueError("outlier_fraction must be in [0, 1)")
        if rank1_power_iterations <= 0:
            raise ValueError("rank1_power_iterations must be positive")
        if block_transform not in {"none", "randomized_hadamard"}:
            raise ValueError(
                "block_transform must be 'none' or 'randomized_hadamard'"
            )
        if allocation not in {"per_block", "optimizer_aware_global"}:
            raise ValueError(
                "allocation must be 'per_block' or 'optimizer_aware_global'"
            )
        if not 0 < preconditioned_relative_squared_error < 1:
            raise ValueError(
                "preconditioned_relative_squared_error must be in (0, 1)"
            )
        if not 0 <= sensitivity_beta2 < 1:
            raise ValueError("sensitivity_beta2 must be in [0, 1)")
        if sensitivity_epsilon <= 0:
            raise ValueError("sensitivity_epsilon must be positive")
        if allocator_iterations <= 0:
            raise ValueError("allocator_iterations must be positive")
        if allocation == "optimizer_aware_global" and (
            prediction != "zero"
            or block_transform != "randomized_hadamard"
            or outlier_fraction != 0.0
        ):
            raise ValueError(
                "optimizer_aware_global requires zero prediction, randomized "
                "Hadamard blocks, and no sparse outlier mode"
            )
        self.block_size = block_size
        self.relative_squared_error = relative_squared_error
        self.prediction = prediction
        self.outlier_fraction = outlier_fraction
        self.rank1_power_iterations = rank1_power_iterations
        self.rank1_warm_start = rank1_warm_start
        self.block_transform = block_transform
        self.allocation = allocation
        self.preconditioned_relative_squared_error = (
            preconditioned_relative_squared_error
        )
        self.sensitivity_beta2 = sensitivity_beta2
        self.sensitivity_epsilon = sensitivity_epsilon
        self.allocator_iterations = allocator_iterations
        self._rht_sign_cache: Dict[int, torch.Tensor] = {}
        self._last_allocator_metrics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        if self.allocation == "optimizer_aware_global":
            return "ots_rht_adam_allocator_f2"
        if self.block_transform == "randomized_hadamard":
            return "ots_rht_deltaq_f1"
        if self.prediction == "rank1_tensor":
            return (
                "ots_rank1_tracking_e2"
                if self.rank1_warm_start
                else "ots_rank1_deltaq_e1"
            )
        if self.outlier_fraction:
            return "ots_deltaq_outlier_v1"
        if self.prediction == "zero":
            return "ots_deltaq_zero_v1"
        if self.prediction == "adaptive_zero_previous":
            return "ots_deltaq_adaptive_predictor_v1"
        return "ots_deltaq_v1"

    def configuration(self) -> dict:
        return {
            "block_size": self.block_size,
            "relative_squared_error": self.relative_squared_error,
            "prediction": self.prediction,
            "int8_rounding": "nearest",
            "fallbacks": ["float16", "float32"],
            "outlier_fraction": self.outlier_fraction,
            "outlier_storage": "int32_index_plus_float16_value",
            "rank1_power_iterations": self.rank1_power_iterations,
            "rank1_factor_dtype": "float16",
            "rank1_warm_start": self.rank1_warm_start,
            "block_transform": self.block_transform,
            "rht_sign_seed": 0x5EED_F101,
            "allocation": self.allocation,
            "preconditioned_relative_squared_error": (
                self.preconditioned_relative_squared_error
            ),
            "sensitivity_beta2": self.sensitivity_beta2,
            "sensitivity_epsilon": self.sensitivity_epsilon,
            "sensitivity_granularity": "one_ema_mean_square_per_block",
            "sensitivity_state_source": "previously_decoded_gradients_only",
            "allocator": "two_multiplier_separable_lagrangian",
            "allocator_iterations": self.allocator_iterations,
        }

    def open_online(
        self,
        artifact_path: Path,
        *,
        source_metadata: Mapping[str, Any],
        predictions_path: Optional[Path] = None,
        prediction_metrics_path: Optional[Path] = None,
        summary_path: Optional[Path] = None,
        durable: bool = False,
        access_mode: str = "online",
    ) -> OTSDeltaQOnlineWriter:
        return OTSDeltaQOnlineWriter(
            self,
            artifact_path,
            source_metadata=source_metadata,
            predictions_path=predictions_path,
            prediction_metrics_path=prediction_metrics_path,
            summary_path=summary_path,
            durable=durable,
            access_mode=access_mode,
        )

    def compress(
        self,
        trace_path: Path,
        artifact_path: Path,
        *,
        predictions_path: Optional[Path] = None,
        prediction_metrics_path: Optional[Path] = None,
    ) -> DeltaQStats:
        trace_path, artifact_path = Path(trace_path), Path(artifact_path)
        reader = GradientTraceReader(trace_path)
        with self.open_online(
            artifact_path,
            source_metadata=reader.metadata,
            predictions_path=predictions_path,
            prediction_metrics_path=prediction_metrics_path,
            access_mode="causal_posthoc",
        ) as writer:
            for record in reader.steps():
                writer.append(record.step, record.gradients)
        return DeltaQStats(**writer._counters)

    def decompress(self, artifact_path: Path, trace_path: Path) -> None:
        artifact_path, trace_path = Path(artifact_path), Path(trace_path)
        predictors: Dict[str, torch.Tensor] = {}
        sensitivity_state: Dict[str, torch.Tensor] = {}
        with artifact_path.open("rb") as stream:
            header = _read_header(stream)
            source_metadata = header["source_metadata"]
            with GradientTraceWriter(trace_path, experiment_id=str(source_metadata["experiment_id"]), metadata={key: value for key, value in source_metadata.items() if key not in {"experiment_id", "format"}}) as writer:
                previous = -1
                while True:
                    fixed = stream.read(_RECORD_HEADER.size)
                    if not fixed:
                        return
                    if len(fixed) != _RECORD_HEADER.size:
                        raise ValueError("truncated OTS-DeltaQ record header")
                    magic, step, metadata_size, payload_size = _RECORD_HEADER.unpack(fixed)
                    if magic != _RECORD_MAGIC or step <= previous:
                        raise ValueError("invalid OTS-DeltaQ record")
                    metadata = json.loads(_read_exact(stream, metadata_size, "record metadata"))
                    payload = _read_exact(stream, payload_size, "record payload")
                    gradients, decoded, _ = self._decode_step(metadata, payload, predictors)
                    writer.append(step, gradients)
                    predictors.update(decoded)
                    if self.allocation == "optimizer_aware_global":
                        self._update_sensitivity_state(decoded, sensitivity_state)
                    previous = step

    def extract_predictions(
        self, artifact_path: Path, predictions_path: Path
    ) -> None:
        """Recreate and store every decoder-side prediction without re-encoding."""

        predictors: Dict[str, torch.Tensor] = {}
        sensitivity_state: Dict[str, torch.Tensor] = {}
        with Path(artifact_path).open("rb") as stream:
            header = _read_header(stream)
            source_metadata = header["source_metadata"]
            extra_metadata = {
                key: value
                for key, value in source_metadata.items()
                if key not in {"experiment_id", "format"}
            }
            extra_metadata["artifact_role"] = "decoder_predictions"
            with GradientTraceWriter(
                Path(predictions_path),
                experiment_id=str(source_metadata["experiment_id"]),
                metadata=extra_metadata,
            ) as writer:
                previous = -1
                while True:
                    fixed = stream.read(_RECORD_HEADER.size)
                    if not fixed:
                        return
                    if len(fixed) != _RECORD_HEADER.size:
                        raise ValueError("truncated OTS-DeltaQ record header")
                    magic, step, metadata_size, payload_size = _RECORD_HEADER.unpack(
                        fixed
                    )
                    if magic != _RECORD_MAGIC or step <= previous:
                        raise ValueError("invalid OTS-DeltaQ record")
                    metadata = json.loads(
                        _read_exact(stream, metadata_size, "record metadata")
                    )
                    payload = _read_exact(stream, payload_size, "record payload")
                    _, decoded, predictions = self._decode_step(
                        metadata, payload, predictors
                    )
                    writer.append(step, predictions)
                    predictors.update(decoded)
                    if self.allocation == "optimizer_aware_global":
                        self._update_sensitivity_state(decoded, sensitivity_state)
                    previous = step

    def inspect_artifact(self, artifact_path: Path) -> DeltaQStats:
        """Count encoded modes by block and element without decoding values."""

        counters = {
            "steps": 0,
            "tensors": 0,
            "int8_blocks": 0,
            "float16_blocks": 0,
            "float32_blocks": 0,
            "int8_elements": 0,
            "float16_elements": 0,
            "float32_elements": 0,
            "previous_prediction_blocks": 0,
            "zero_prediction_blocks": 0,
            "previous_prediction_elements": 0,
            "zero_prediction_elements": 0,
            "int8_outlier_blocks": 0,
            "int8_outlier_elements": 0,
            "outlier_values": 0,
            "rank1_prediction_tensors": 0,
            "rank1_prediction_blocks": 0,
            "rank1_prediction_elements": 0,
            "rank1_factor_bytes": 0,
            "rht_int8_blocks": 0,
            "rht_int8_elements": 0,
        }
        with Path(artifact_path).open("rb") as stream:
            header = _read_header(stream)
            configuration = header.get("configuration", {})
            if int(configuration.get("block_size", -1)) != self.block_size:
                raise ValueError("artifact block size does not match compressor")
            while True:
                fixed = stream.read(_RECORD_HEADER.size)
                if not fixed:
                    return DeltaQStats(**counters)
                if len(fixed) != _RECORD_HEADER.size:
                    raise ValueError("truncated OTS-DeltaQ record header")
                magic, _, metadata_size, payload_size = _RECORD_HEADER.unpack(fixed)
                if magic != _RECORD_MAGIC:
                    raise ValueError("invalid OTS-DeltaQ record")
                metadata = json.loads(
                    _read_exact(stream, metadata_size, "record metadata")
                )
                payload = _read_exact(stream, payload_size, "record payload")
                counters["steps"] += 1
                for descriptor in metadata["tensors"]:
                    if not descriptor.get("present", False):
                        continue
                    counters["tensors"] += 1
                    elements = math.prod(descriptor["shape"])
                    cursor = int(descriptor["offset"])
                    end = cursor + int(descriptor["nbytes"])
                    predictor_label = str(descriptor.get("prediction", ""))
                    if predictor_label == "rank1":
                        factor_bytes = int(descriptor["rank1_factor_nbytes"])
                        counters["rank1_prediction_tensors"] += 1
                        counters["rank1_factor_bytes"] += factor_bytes
                        cursor += factor_bytes
                    remaining = elements
                    while remaining:
                        count = min(self.block_size, remaining)
                        code, _ = _BLOCK_HEADER.unpack_from(payload, cursor)
                        cursor += _BLOCK_HEADER.size
                        mode = code & 0x03
                        if mode == _INT8_OUTLIER:
                            exception_count = _OUTLIER_COUNT.unpack_from(
                                payload, cursor
                            )[0]
                            cursor += _OUTLIER_COUNT.size
                            width = 1
                            label = "int8_outlier"
                            counters["outlier_values"] += exception_count
                            exception_bytes = 6 * exception_count
                        else:
                            width = {
                                _INT8: 1,
                                _FLOAT16: 2,
                                _FLOAT32: 4,
                            }.get(mode)
                            if width is None:
                                raise ValueError("unknown OTS-DeltaQ block mode")
                            label = {
                                _INT8: "int8",
                                _FLOAT16: "float16",
                                _FLOAT32: "float32",
                            }[mode]
                            exception_bytes = 0
                        counters[f"{label}_blocks"] += 1
                        counters[f"{label}_elements"] += count
                        if mode == _INT8 and code & _RHT_FLAG:
                            counters["rht_int8_blocks"] += 1
                            counters["rht_int8_elements"] += count
                        if predictor_label == "rank1":
                            predictor_counter = "rank1_prediction"
                        elif configuration.get("prediction") == "zero":
                            predictor_counter = "zero_prediction"
                        elif configuration.get("prediction") == "adaptive_zero_previous":
                            predictor_counter = "zero_prediction" if code & 0x04 else "previous_prediction"
                        else:
                            predictor_counter = "previous_prediction"
                        counters[f"{predictor_counter}_blocks"] += 1
                        counters[f"{predictor_counter}_elements"] += count
                        cursor += count * width + exception_bytes
                        remaining -= count
                    if cursor != end:
                        raise ValueError("invalid OTS-DeltaQ tensor extent")

    def _encode_step(
        self,
        record: GradientStep,
        predictors: Mapping[str, torch.Tensor],
        counters: dict,
        rank1_vectors: Optional[Dict[str, torch.Tensor]] = None,
        sensitivity_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        bytes,
        bytes,
        Dict[str, torch.Tensor],
        Dict[str, Optional[torch.Tensor]],
    ]:
        self._last_allocator_metrics = {}
        if self.allocation == "optimizer_aware_global":
            if sensitivity_state is None:
                sensitivity_state = {}
            return self._encode_step_optimizer_aware(
                record, counters, sensitivity_state
            )
        descriptors, payload_parts, decoded, selected_predictions = [], [], {}, {}
        offset = 0
        for name, tensor in record.gradients.items():
            if tensor is None:
                descriptors.append({"name": name, "present": False})
                selected_predictions[name] = None
                continue
            value = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
            prediction = self._prediction_for(name, value, predictors)
            base_kind = "zero" if self.prediction == "zero" else "previous"
            encoded, reconstruction, selected_prediction, block_counts = (
                self._encode_tensor(value, prediction, predictor_kind=base_kind)
            )
            descriptor_extra = {}
            if self.prediction == "rank1_tensor" and value.ndim == 2:
                initial_right = None
                if self.rank1_warm_start and rank1_vectors is not None:
                    initial_right = rank1_vectors.get(name)
                rank1 = self._encode_rank1_prediction(value, initial_right)
                if rank1 is not None:
                    rank1_prediction, factors, tracked_right = rank1
                    if self.rank1_warm_start and rank1_vectors is not None:
                        rank1_vectors[name] = tracked_right
                    rank1_encoded, rank1_reconstruction, _, rank1_counts = (
                        self._encode_tensor(
                            value, rank1_prediction, predictor_kind="rank1"
                        )
                    )
                    if len(factors) + len(rank1_encoded) < len(encoded):
                        encoded = factors + rank1_encoded
                        reconstruction = rank1_reconstruction
                        selected_prediction = rank1_prediction
                        block_counts = rank1_counts
                        block_counts["rank1_prediction_tensors"] = 1
                        block_counts["rank1_factor_bytes"] = len(factors)
                        descriptor_extra = {
                            "prediction": "rank1",
                            "rank1_factor_nbytes": len(factors),
                        }
            counters["tensors"] += 1
            for key, amount in block_counts.items():
                counters[key] += amount
            descriptors.append({"name": name, "present": True, "dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape), "offset": offset, "nbytes": len(encoded), **descriptor_extra})
            payload_parts.append(encoded)
            offset += len(encoded)
            decoded[name] = reconstruction.reshape(value.shape)
            selected_predictions[name] = selected_prediction.reshape(value.shape)
        return (
            _canonical_json({"tensors": descriptors}),
            b"".join(payload_parts),
            decoded,
            selected_predictions,
        )

    def _encode_step_optimizer_aware(
        self,
        record: GradientStep,
        counters: dict,
        sensitivity_state: Dict[str, torch.Tensor],
    ) -> Tuple[
        bytes,
        bytes,
        Dict[str, torch.Tensor],
        Dict[str, Optional[torch.Tensor]],
    ]:
        """Build all block frontiers, then allocate precision across the step.

        The state contains one EMA mean-square value per block and is updated
        from the selected reconstruction.  An independent decoder can therefore
        reproduce it exactly without the original gradients.
        """

        values: Dict[str, torch.Tensor] = {}
        dtypes: Dict[str, torch.dtype] = {}
        shapes: Dict[str, Tuple[int, ...]] = {}
        tensor_blocks: Dict[str, List[int]] = {}
        blocks: List[_F2Block] = []
        absent: List[str] = []

        for name, tensor in record.gradients.items():
            if tensor is None:
                absent.append(name)
                continue
            value = (
                tensor.detach()
                .to(dtype=torch.float32, device="cpu")
                .contiguous()
            )
            values[name] = value
            dtypes[name] = tensor.dtype
            shapes[name] = tuple(tensor.shape)
            tensor_blocks[name] = []
            flat = value.reshape(-1)
            previous_sensitivity = sensitivity_state.get(name)
            block_count = math.ceil(flat.numel() / self.block_size)
            if (
                previous_sensitivity is None
                or previous_sensitivity.numel() != block_count
            ):
                previous_sensitivity = None
            for block_index, offset in enumerate(
                range(0, flat.numel(), self.block_size)
            ):
                current = flat[offset : offset + self.block_size]
                energy = float(torch.sum(current.square(), dtype=torch.float64))
                if previous_sensitivity is None:
                    sensitivity_weight = 1.0
                else:
                    mean_square = max(
                        float(previous_sensitivity[block_index]), 0.0
                    )
                    denominator = math.sqrt(mean_square) + self.sensitivity_epsilon
                    sensitivity_weight = 1.0 / (denominator * denominator)
                tensor_blocks[name].append(len(blocks))
                blocks.append(
                    _F2Block(
                        tensor_name=name,
                        offset=offset,
                        count=current.numel(),
                        energy=energy,
                        sensitivity_weight=sensitivity_weight,
                        candidates=self._f2_candidate_frontier(current),
                    )
                )

        selected, allocator_metrics = self._allocate_f2_candidates(blocks)
        self._last_allocator_metrics = allocator_metrics

        descriptors: List[dict] = []
        payload_parts: List[bytes] = []
        decoded: Dict[str, torch.Tensor] = {}
        predictions: Dict[str, Optional[torch.Tensor]] = {
            name: None for name in absent
        }
        payload_offset = 0
        for name, tensor in record.gradients.items():
            if tensor is None:
                descriptors.append({"name": name, "present": False})
                continue
            value = values[name]
            reconstruction = torch.empty_like(value.reshape(-1))
            output = io.BytesIO()
            for global_index in tensor_blocks[name]:
                block = blocks[global_index]
                candidate = block.candidates[selected[global_index]]
                code = _INT8 if candidate.mode == _INT8 else candidate.mode
                code |= 1 << 2
                if candidate.transformed:
                    code |= _RHT_FLAG
                output.write(_BLOCK_HEADER.pack(code, candidate.scale))
                output.write(candidate.encoded)
                reconstruction[
                    block.offset : block.offset + block.count
                ] = candidate.restored
                label = {
                    _INT8: "int8",
                    _FLOAT16: "float16",
                    _FLOAT32: "float32",
                }[candidate.mode]
                counters[f"{label}_blocks"] += 1
                counters[f"{label}_elements"] += block.count
                counters["zero_prediction_blocks"] += 1
                counters["zero_prediction_elements"] += block.count
                if candidate.transformed:
                    counters["rht_int8_blocks"] += 1
                    counters["rht_int8_elements"] += block.count
            encoded = output.getvalue()
            counters["tensors"] += 1
            descriptors.append(
                {
                    "name": name,
                    "present": True,
                    "dtype": str(dtypes[name]).removeprefix("torch."),
                    "shape": list(shapes[name]),
                    "offset": payload_offset,
                    "nbytes": len(encoded),
                }
            )
            payload_parts.append(encoded)
            payload_offset += len(encoded)
            restored_tensor = reconstruction.reshape(shapes[name])
            decoded[name] = restored_tensor
            predictions[name] = torch.zeros_like(restored_tensor)

        self._update_sensitivity_state(decoded, sensitivity_state)
        return (
            _canonical_json({"tensors": descriptors}),
            b"".join(payload_parts),
            decoded,
            predictions,
        )

    def _f2_candidate_frontier(
        self, value: torch.Tensor
    ) -> Tuple[_F2Candidate, ...]:
        """Return the non-dominated F1-int8/FP16/FP32 frontier."""

        transformed = (
            value.numel() >= 2
            and value.numel() & (value.numel() - 1) == 0
        )
        quantizer_input = (
            self._randomized_hadamard(value) if transformed else value
        )
        maximum = float(quantizer_input.abs().max()) if value.numel() else 0.0
        scale = maximum / 127.0 if maximum else 1.0
        # The scale is serialized as float32.  Score the exact value the decoder
        # sees, rather than a higher precision Python scalar.
        scale = struct.unpack("<f", struct.pack("<f", scale))[0]
        quantized = (
            torch.round(quantizer_input / scale)
            .clamp(-127, 127)
            .to(torch.int8)
        )
        int8_restored = quantized.to(torch.float32) * scale
        if transformed:
            int8_restored = self._inverse_randomized_hadamard(int8_restored)
        half = value.to(torch.float16)
        half_restored = half.to(torch.float32)
        candidates = (
            _F2Candidate(
                mode=_INT8,
                scale=scale,
                encoded=quantized.numpy().tobytes(),
                restored=int8_restored,
                squared_error=float(
                    torch.sum(
                        (value - int8_restored).square(), dtype=torch.float64
                    )
                ),
                transformed=transformed,
            ),
            _F2Candidate(
                mode=_FLOAT16,
                scale=1.0,
                encoded=half.numpy().tobytes(),
                restored=half_restored,
                squared_error=float(
                    torch.sum(
                        (value - half_restored).square(), dtype=torch.float64
                    )
                ),
                transformed=False,
            ),
            _F2Candidate(
                mode=_FLOAT32,
                scale=1.0,
                encoded=value.numpy().tobytes(),
                restored=value,
                squared_error=0.0,
                transformed=False,
            ),
        )
        frontier: List[_F2Candidate] = []
        best_error = math.inf
        for candidate in sorted(
            candidates, key=lambda item: (item.payload_nbytes, item.squared_error)
        ):
            if candidate.squared_error < best_error:
                frontier.append(candidate)
                best_error = candidate.squared_error
        return tuple(frontier)

    def _allocate_f2_candidates(
        self, blocks: Sequence[_F2Block]
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Solve F2's separable two-constraint Lagrangian approximately.

        For fixed multipliers every block choice is independent, so one pass is
        linear in the number of blocks and frontier points.  Multiplicative dual
        updates find a feasible discrete solution; a final feasibility-preserving
        downgrade pass removes avoidable bytes.
        """

        if not blocks:
            return [], {
                "allocator_raw_relative_squared_error": 0.0,
                "optimizer_proxy_relative_squared_error": 0.0,
                "allocator_dual_iterations": 0,
                "allocator_selected_payload_bytes": 0,
            }
        source_bytes = max(sum(4 * block.count for block in blocks), 1)
        raw_energy = max(sum(block.energy for block in blocks), 1e-30)
        weighted_energy = max(
            sum(block.energy * block.sensitivity_weight for block in blocks),
            1e-30,
        )

        def choose(lambda_raw: float, lambda_weighted: float) -> List[int]:
            result: List[int] = []
            for block in blocks:
                result.append(
                    min(
                        range(len(block.candidates)),
                        key=lambda index: (
                            block.candidates[index].payload_nbytes / source_bytes
                            + lambda_raw
                            * block.candidates[index].squared_error
                            / raw_energy
                            + lambda_weighted
                            * block.candidates[index].squared_error
                            * block.sensitivity_weight
                            / weighted_energy,
                            block.candidates[index].payload_nbytes,
                            block.candidates[index].squared_error,
                        ),
                    )
                )
            return result

        def totals(selection: Sequence[int]) -> Tuple[int, float, float]:
            payload_bytes = 0
            raw_error = weighted_error = 0.0
            for block, candidate_index in zip(blocks, selection):
                candidate = block.candidates[candidate_index]
                payload_bytes += candidate.payload_nbytes
                raw_error += candidate.squared_error
                weighted_error += (
                    candidate.squared_error * block.sensitivity_weight
                )
            return payload_bytes, raw_error, weighted_error

        raw_limit = self.relative_squared_error * raw_energy
        weighted_limit = (
            self.preconditioned_relative_squared_error * weighted_energy
        )
        lambda_raw = 1.0 / self.relative_squared_error
        lambda_weighted = 1.0 / self.preconditioned_relative_squared_error
        feasible: List[Tuple[int, float, float, List[int]]] = []

        # The exact point guarantees that a feasible discrete solution always
        # exists, independently of dual convergence.
        exact = [
            min(
                range(len(block.candidates)),
                key=lambda index: block.candidates[index].squared_error,
            )
            for block in blocks
        ]
        exact_bytes, exact_raw, exact_weighted = totals(exact)
        feasible.append((exact_bytes, exact_raw, exact_weighted, exact))

        for _ in range(self.allocator_iterations):
            selection = choose(lambda_raw, lambda_weighted)
            payload_bytes, raw_error, weighted_error = totals(selection)
            if raw_error <= raw_limit and weighted_error <= weighted_limit:
                feasible.append(
                    (payload_bytes, raw_error, weighted_error, selection)
                )
            raw_ratio = raw_error / max(raw_limit, 1e-30)
            weighted_ratio = weighted_error / max(weighted_limit, 1e-30)
            lambda_raw *= math.exp(
                max(-2.0, min(2.0, 0.65 * (raw_ratio - 1.0)))
            )
            lambda_weighted *= math.exp(
                max(-2.0, min(2.0, 0.65 * (weighted_ratio - 1.0)))
            )
            lambda_raw = max(lambda_raw, 1e-12)
            lambda_weighted = max(lambda_weighted, 1e-12)

        payload_bytes, raw_error, weighted_error, selected = min(
            feasible,
            key=lambda item: (item[0], item[1] + item[2]),
        )
        selected = list(selected)

        # Try all cheaper one-block moves in decreasing byte-saving order.  This
        # is a local cleanup after the non-greedy dual allocation, not the mode
        # selection mechanism itself.
        proposals: List[Tuple[int, int, int, float, float]] = []
        for block_index, (block, chosen_index) in enumerate(zip(blocks, selected)):
            chosen = block.candidates[chosen_index]
            for candidate_index, candidate in enumerate(block.candidates):
                saved = chosen.payload_nbytes - candidate.payload_nbytes
                if saved <= 0:
                    continue
                proposals.append(
                    (
                        saved,
                        block_index,
                        candidate_index,
                        candidate.squared_error - chosen.squared_error,
                        (
                            candidate.squared_error - chosen.squared_error
                        )
                        * block.sensitivity_weight,
                    )
                )
        downgrades = 0
        for saved, block_index, candidate_index, raw_delta, weighted_delta in sorted(
            proposals, key=lambda item: (-item[0], item[3] + item[4])
        ):
            block = blocks[block_index]
            current = block.candidates[selected[block_index]]
            candidate = block.candidates[candidate_index]
            if candidate.payload_nbytes >= current.payload_nbytes:
                continue
            actual_raw_delta = candidate.squared_error - current.squared_error
            actual_weighted_delta = actual_raw_delta * block.sensitivity_weight
            if (
                raw_error + actual_raw_delta <= raw_limit
                and weighted_error + actual_weighted_delta <= weighted_limit
            ):
                selected[block_index] = candidate_index
                payload_bytes -= current.payload_nbytes - candidate.payload_nbytes
                raw_error += actual_raw_delta
                weighted_error += actual_weighted_delta
                downgrades += 1

        weights = [block.sensitivity_weight for block in blocks]
        return selected, {
            "allocator_raw_relative_squared_error": raw_error / raw_energy,
            "optimizer_proxy_relative_squared_error": (
                weighted_error / weighted_energy
            ),
            "allocator_dual_iterations": self.allocator_iterations,
            "allocator_local_downgrades": downgrades,
            "allocator_selected_payload_bytes": payload_bytes,
            "allocator_block_count": len(blocks),
            "sensitivity_weight_min": min(weights),
            "sensitivity_weight_max": max(weights),
        }

    def _update_sensitivity_state(
        self,
        decoded: Mapping[str, torch.Tensor],
        sensitivity_state: Dict[str, torch.Tensor],
    ) -> None:
        """Advance coarse Adam-v state using decoded values only."""

        active = set(decoded)
        for stale in set(sensitivity_state) - active:
            del sensitivity_state[stale]
        for name, value in decoded.items():
            flat = value.reshape(-1)
            mean_squares = torch.tensor(
                [
                    float(
                        torch.mean(
                            flat[offset : offset + self.block_size].square(),
                            dtype=torch.float64,
                        )
                    )
                    for offset in range(0, flat.numel(), self.block_size)
                ],
                dtype=torch.float64,
            )
            previous = sensitivity_state.get(name)
            if previous is None or previous.shape != mean_squares.shape:
                previous = torch.zeros_like(mean_squares)
            sensitivity_state[name] = (
                self.sensitivity_beta2 * previous
                + (1.0 - self.sensitivity_beta2) * mean_squares
            )

    def _decode_step(self, metadata: Mapping[str, object], payload: bytes, predictors: Mapping[str, torch.Tensor]) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        gradients: Dict[str, Optional[torch.Tensor]] = {}
        decoded: Dict[str, torch.Tensor] = {}
        selected_predictions: Dict[str, Optional[torch.Tensor]] = {}
        for descriptor in metadata["tensors"]:  # type: ignore[index]
            name = descriptor["name"]  # type: ignore[index]
            if not descriptor.get("present", False):  # type: ignore[union-attr]
                gradients[name] = None
                selected_predictions[name] = None
                continue
            shape = tuple(descriptor["shape"])  # type: ignore[index]
            prediction = self._prediction_for(
                name, torch.zeros(shape, dtype=torch.float32), predictors
            )
            offset, nbytes = int(descriptor["offset"]), int(descriptor["nbytes"])
            encoded = payload[offset : offset + nbytes]
            if descriptor.get("prediction") == "rank1":  # type: ignore[union-attr]
                factor_nbytes = int(descriptor["rank1_factor_nbytes"])  # type: ignore[index]
                prediction = self._decode_rank1_prediction(encoded[:factor_nbytes], shape)
                encoded = encoded[factor_nbytes:]
            reconstructed = self._decode_tensor(
                encoded, prediction
            )
            dtype = getattr(torch, str(descriptor["dtype"]))
            gradients[name] = reconstructed.to(dtype=dtype).reshape(shape)
            decoded[name] = reconstructed.reshape(shape)
            selected_predictions[name] = prediction.reshape(shape)
        return gradients, decoded, selected_predictions

    def _prediction_for(
        self,
        name: str,
        value: torch.Tensor,
        predictors: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.prediction == "zero":
            return torch.zeros_like(value, dtype=torch.float32, device="cpu")
        prediction = predictors.get(name)
        if prediction is None or prediction.shape != value.shape:
            return torch.zeros_like(value, dtype=torch.float32, device="cpu")
        return prediction

    def _encode_tensor(
        self,
        value: torch.Tensor,
        prediction: torch.Tensor,
        *,
        predictor_kind: Optional[str] = None,
    ) -> Tuple[bytes, torch.Tensor, torch.Tensor, dict]:
        source, previous = value.reshape(-1), prediction.reshape(-1)
        output = io.BytesIO()
        reconstruction = torch.empty_like(source)
        selected_prediction = torch.empty_like(source)
        counts = {
            "int8_blocks": 0,
            "float16_blocks": 0,
            "float32_blocks": 0,
            "int8_elements": 0,
            "float16_elements": 0,
            "float32_elements": 0,
            "previous_prediction_blocks": 0,
            "zero_prediction_blocks": 0,
            "previous_prediction_elements": 0,
            "zero_prediction_elements": 0,
            "int8_outlier_blocks": 0,
            "int8_outlier_elements": 0,
            "outlier_values": 0,
            "rank1_prediction_tensors": 0,
            "rank1_prediction_blocks": 0,
            "rank1_prediction_elements": 0,
            "rank1_factor_bytes": 0,
            "rht_int8_blocks": 0,
            "rht_int8_elements": 0,
        }
        if predictor_kind is None:
            predictor_kind = "zero" if self.prediction == "zero" else "previous"
        for offset in range(0, source.numel(), self.block_size):
            current = source[offset : offset + self.block_size]
            previous_block = previous[offset : offset + self.block_size]
            if self.prediction == "adaptive_zero_previous":
                candidates = [
                    self._encode_candidate(current, previous_block, 0),
                    self._encode_candidate(current, torch.zeros_like(current), 1),
                ]
                (
                    mode,
                    scale,
                    encoded,
                    restored,
                    error,
                    predictor_id,
                    predicted,
                    transformed,
                ) = min(
                    candidates, key=lambda candidate: (len(candidate[2]), candidate[4])
                )
            else:
                predictor_id = 1 if predictor_kind == "zero" else 0
                predicted = previous_block
                (
                    mode,
                    scale,
                    encoded,
                    restored,
                    error,
                    _,
                    _,
                    transformed,
                ) = self._encode_candidate(current, predicted, predictor_id)
            del error
            code = mode | (predictor_id << 2) | (_RHT_FLAG if transformed else 0)
            output.write(_BLOCK_HEADER.pack(code, scale))
            output.write(encoded)
            reconstruction[offset : offset + current.numel()] = predicted + restored
            selected_prediction[offset : offset + current.numel()] = predicted
            label = {
                _INT8: "int8",
                _FLOAT16: "float16",
                _FLOAT32: "float32",
                _INT8_OUTLIER: "int8_outlier",
            }[mode]
            counts[f"{label}_blocks"] += 1
            counts[f"{label}_elements"] += current.numel()
            if mode == _INT8_OUTLIER:
                counts["outlier_values"] += _OUTLIER_COUNT.unpack_from(
                    encoded, 0
                )[0]
            if transformed:
                counts["rht_int8_blocks"] += 1
                counts["rht_int8_elements"] += current.numel()
            predictor_label = (
                "rank1_prediction"
                if predictor_kind == "rank1"
                else ("zero_prediction" if predictor_id else "previous_prediction")
            )
            counts[f"{predictor_label}_blocks"] += 1
            counts[f"{predictor_label}_elements"] += current.numel()
        return output.getvalue(), reconstruction, selected_prediction, counts

    def _encode_rank1_prediction(
        self,
        value: torch.Tensor,
        initial_right: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, bytes, torch.Tensor]]:
        rows, columns = value.shape
        if rows < 2 or columns < 2:
            return None
        if initial_right is None or initial_right.numel() != columns:
            right = torch.ones(columns, dtype=torch.float32) / math.sqrt(columns)
        else:
            right = initial_right.detach().to(torch.float32).reshape(columns).clone()
            norm = float(torch.linalg.vector_norm(right))
            if norm == 0.0:
                right.fill_(1.0 / math.sqrt(columns))
            else:
                right /= norm
        for _ in range(self.rank1_power_iterations):
            left = value.mv(right)
            right = value.T.mv(left)
            norm = float(torch.linalg.vector_norm(right))
            if norm == 0.0:
                return None
            right /= norm
        left = value.mv(right)
        left_half, right_half = left.to(torch.float16), right.to(torch.float16)
        decoded_left = left_half.to(torch.float32)
        decoded_right = right_half.to(torch.float32)
        prediction = torch.outer(decoded_left, decoded_right)
        factors = left_half.numpy().tobytes() + right_half.numpy().tobytes()
        return prediction, factors, right.clone()

    @staticmethod
    def _decode_rank1_prediction(
        payload: bytes, shape: Tuple[int, ...]
    ) -> torch.Tensor:
        if len(shape) != 2:
            raise ValueError("rank1 prediction requires a matrix tensor")
        rows, columns = shape
        expected = 2 * (rows + columns)
        if len(payload) != expected:
            raise ValueError("invalid rank1 factor payload")
        factors = torch.frombuffer(bytearray(payload), dtype=torch.float16)
        left = factors[:rows].to(torch.float32)
        right = factors[rows:].to(torch.float32)
        return torch.outer(left, right)

    def _encode_candidate(
        self,
        value: torch.Tensor,
        prediction: torch.Tensor,
        predictor_id: int,
    ) -> Tuple[
        int,
        float,
        bytes,
        torch.Tensor,
        float,
        int,
        torch.Tensor,
        bool,
    ]:
        residual = value - prediction
        mode, scale, encoded, restored, transformed = self._encode_block(
            value, residual
        )
        error = float(torch.sum((residual - restored).square(), dtype=torch.float64))
        if self.outlier_fraction:
            outlier = self._encode_outlier_block(value, residual)
            if outlier is not None:
                outlier_mode, outlier_scale, outlier_encoded, outlier_restored = outlier
                outlier_error = float(
                    torch.sum(
                        (residual - outlier_restored).square(), dtype=torch.float64
                    )
                )
                if (len(outlier_encoded), outlier_error) < (len(encoded), error):
                    mode = outlier_mode
                    scale = outlier_scale
                    encoded = outlier_encoded
                    restored = outlier_restored
                    error = outlier_error
                    transformed = False
        return (
            mode,
            scale,
            encoded,
            restored,
            error,
            predictor_id,
            prediction,
            transformed,
        )

    def _encode_outlier_block(
        self, value: torch.Tensor, residual: torch.Tensor
    ) -> Optional[Tuple[int, float, bytes, torch.Tensor]]:
        count = residual.numel()
        exception_count = max(1, math.ceil(count * self.outlier_fraction))
        if count == 0 or exception_count >= count:
            return None
        exception_indices = torch.topk(
            residual.abs(), exception_count, sorted=False
        ).indices
        inlier_residual = residual.clone()
        inlier_residual[exception_indices] = 0.0
        maximum = float(inlier_residual.abs().max())
        scale = maximum / 127.0 if maximum else 1.0
        quantized = torch.round(inlier_residual / scale).clamp(-127, 127).to(
            torch.int8
        )
        exceptions = residual[exception_indices].to(torch.float16)
        restored = quantized.to(torch.float32) * scale
        restored[exception_indices] = exceptions.to(torch.float32)
        energy = float(torch.sum(value.square(), dtype=torch.float64))
        if not _within_budget(
            value, residual - restored, energy, self.relative_squared_error
        ):
            return None
        encoded = b"".join(
            (
                _OUTLIER_COUNT.pack(exception_count),
                quantized.numpy().tobytes(),
                exception_indices.to(torch.int32).numpy().tobytes(),
                exceptions.numpy().tobytes(),
            )
        )
        return _INT8_OUTLIER, scale, encoded, restored

    def _encode_block(
        self, value: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[int, float, bytes, torch.Tensor, bool]:
        energy = float(torch.sum(value.square(), dtype=torch.float64))
        transformed = False
        quantizer_input = residual
        if (
            self.block_transform == "randomized_hadamard"
            and residual.numel() >= 2
            and residual.numel() & (residual.numel() - 1) == 0
        ):
            quantizer_input = self._randomized_hadamard(residual)
            transformed = True
        maximum = (
            float(quantizer_input.abs().max()) if quantizer_input.numel() else 0.0
        )
        scale = maximum / 127.0 if maximum else 1.0
        quantized = (
            torch.round(quantizer_input / scale).clamp(-127, 127).to(torch.int8)
        )
        restored = quantized.to(torch.float32) * scale
        if transformed:
            restored = self._inverse_randomized_hadamard(restored)
        if _within_budget(value, residual - restored, energy, self.relative_squared_error):
            return _INT8, scale, quantized.numpy().tobytes(), restored, transformed
        half = residual.to(torch.float16)
        restored = half.to(torch.float32)
        if _within_budget(value, residual - restored, energy, self.relative_squared_error):
            return _FLOAT16, 1.0, half.numpy().tobytes(), restored, False
        return _FLOAT32, 1.0, residual.numpy().tobytes(), residual, False

    def _randomized_hadamard(self, value: torch.Tensor) -> torch.Tensor:
        signs = self._rht_signs(value.numel())
        return _fast_walsh_hadamard(value * signs)

    def _inverse_randomized_hadamard(self, value: torch.Tensor) -> torch.Tensor:
        signs = self._rht_signs(value.numel())
        return _fast_walsh_hadamard(value) * signs

    def _rht_signs(self, count: int) -> torch.Tensor:
        signs = self._rht_sign_cache.get(count)
        if signs is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(0x5EED_F101 + count)
            signs = torch.randint(
                0, 2, (count,), generator=generator, dtype=torch.int8
            ).to(torch.float32)
            signs.mul_(2.0).sub_(1.0)
            self._rht_sign_cache[count] = signs
        return signs

    def _decode_tensor(self, payload: bytes, prediction: torch.Tensor) -> torch.Tensor:
        source = prediction.reshape(-1)
        output = torch.empty_like(source)
        cursor = 0
        for offset in range(0, source.numel(), self.block_size):
            count = min(self.block_size, source.numel() - offset)
            code, scale = _BLOCK_HEADER.unpack_from(payload, cursor)
            cursor += _BLOCK_HEADER.size
            mode = code & 0x03
            if mode == _INT8_OUTLIER:
                if cursor + _OUTLIER_COUNT.size > len(payload):
                    raise ValueError("invalid OTS-DeltaQ outlier count")
                exception_count = _OUTLIER_COUNT.unpack_from(payload, cursor)[0]
                cursor += _OUTLIER_COUNT.size
                dense_bytes = count
                index_bytes = 4 * exception_count
                value_bytes = 2 * exception_count
                end = cursor + dense_bytes + index_bytes + value_bytes
                if end > len(payload):
                    raise ValueError("invalid OTS-DeltaQ outlier payload")
                dense = payload[cursor : cursor + dense_bytes]
                cursor += dense_bytes
                raw_indices = payload[cursor : cursor + index_bytes]
                cursor += index_bytes
                raw_values = payload[cursor : cursor + value_bytes]
                cursor += value_bytes
                residual = (
                    torch.frombuffer(bytearray(dense), dtype=torch.int8).to(
                        torch.float32
                    )
                    * scale
                )
                indices = torch.frombuffer(
                    bytearray(raw_indices), dtype=torch.int32
                ).to(torch.int64)
                exceptions = torch.frombuffer(
                    bytearray(raw_values), dtype=torch.float16
                ).to(torch.float32)
                if exception_count and int(indices.max()) >= count:
                    raise ValueError("OTS-DeltaQ outlier index is out of range")
                residual[indices] = exceptions
            else:
                byte_count = count * (
                    {_INT8: 1, _FLOAT16: 2, _FLOAT32: 4}.get(mode, 0)
                )
                if not byte_count or cursor + byte_count > len(payload):
                    raise ValueError("invalid OTS-DeltaQ tensor payload")
                raw = payload[cursor : cursor + byte_count]
                cursor += byte_count
            if mode == _INT8:
                residual = torch.frombuffer(bytearray(raw), dtype=torch.int8).to(torch.float32) * scale
                if code & _RHT_FLAG:
                    residual = self._inverse_randomized_hadamard(residual)
            elif mode == _FLOAT16:
                residual = torch.frombuffer(bytearray(raw), dtype=torch.float16).to(torch.float32)
            elif mode == _FLOAT32:
                residual = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
            elif mode != _INT8_OUTLIER:
                raise ValueError("unknown OTS-DeltaQ block mode")
            if self.prediction == "adaptive_zero_previous" and code & 0x04:
                predicted = torch.zeros(count, dtype=torch.float32)
            else:
                predicted = source[offset : offset + count]
            output[offset : offset + count] = predicted + residual
        if cursor != len(payload):
            raise ValueError("unexpected bytes after OTS-DeltaQ tensor")
        return output


def _fast_walsh_hadamard(value: torch.Tensor) -> torch.Tensor:
    """Apply a normalized Walsh-Hadamard transform to a power-of-two vector."""

    count = value.numel()
    if count == 0 or count & (count - 1):
        raise ValueError("Hadamard input length must be a positive power of two")
    output = value.detach().to(torch.float32).reshape(-1).clone()
    width = 1
    while width < count:
        paired = output.reshape(-1, 2, width)
        left = paired[:, 0, :].clone()
        right = paired[:, 1, :].clone()
        paired[:, 0, :] = left + right
        paired[:, 1, :] = left - right
        width *= 2
    return output / math.sqrt(count)


def _within_budget(value: torch.Tensor, error: torch.Tensor, energy: float, budget: float) -> bool:
    squared_error = float(torch.sum(error.square(), dtype=torch.float64))
    return squared_error <= budget * max(energy, 1e-30)


def _step_metrics(
    step: int,
    gradients: Mapping[str, Optional[torch.Tensor]],
    predictions: Mapping[str, Optional[torch.Tensor]],
    decoded: Mapping[str, torch.Tensor],
    *,
    encoded_bytes: int,
    block_counts: Mapping[str, int],
    encode_seconds: float,
    extra_metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    gradient_energy = prediction_energy = prediction_error = 0.0
    prediction_dot = reconstruction_error = 0.0
    elements = 0
    source_tensor_bytes = 0
    for name, tensor in gradients.items():
        if tensor is None:
            continue
        gradient = tensor.detach().to(dtype=torch.float64, device="cpu")
        prediction = predictions[name]
        if prediction is None:
            raise ValueError(f"missing prediction for gradient {name!r}")
        prediction = prediction.to(torch.float64)
        reconstruction = decoded[name].to(torch.float64)
        gradient_energy += float(torch.sum(gradient.square()))
        prediction_energy += float(torch.sum(prediction.square()))
        prediction_dot += float(torch.sum(gradient * prediction))
        prediction_error += float(torch.sum((gradient - prediction).square()))
        reconstruction_error += float(torch.sum((gradient - reconstruction).square()))
        elements += gradient.numel()
        source_tensor_bytes += tensor.numel() * tensor.element_size()
    energy_floor = max(gradient_energy, 1e-30)
    cosine_denominator = (gradient_energy * prediction_energy) ** 0.5
    result = {
        "step": step,
        "gradient_elements": elements,
        "gradient_energy": gradient_energy,
        "prediction_energy": prediction_energy,
        "prediction_squared_error": prediction_error,
        "prediction_energy_r2": 1.0 - prediction_error / energy_floor,
        "prediction_cosine_similarity": (
            prediction_dot / cosine_denominator if cosine_denominator else None
        ),
        "reconstruction_squared_error": reconstruction_error,
        "reconstruction_energy_r2": 1.0 - reconstruction_error / energy_floor,
        "encoded_bytes": encoded_bytes,
        "source_tensor_bytes": source_tensor_bytes,
        "compression_ratio": (
            source_tensor_bytes / encoded_bytes if encoded_bytes else None
        ),
        "compression_latency_ms": 1_000.0 * encode_seconds,
        "compression_throughput_mib_per_second": (
            source_tensor_bytes / (1024.0 * 1024.0 * encode_seconds)
            if encode_seconds
            else None
        ),
        "encoded_bits_per_gradient_element": (
            8.0 * encoded_bytes / elements if elements else 0.0
        ),
        "encode_seconds": encode_seconds,
        **block_counts,
    }
    if extra_metrics:
        result.update(extra_metrics)
    return result


def summarize_prediction_metrics(path: Path) -> Dict[str, Any]:
    """Aggregate plot-ready JSONL without discarding the per-step trajectory."""

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]

    def distribution(field: str) -> Dict[str, float]:
        values = sorted(
            float(row[field]) for row in rows if row.get(field) is not None
        )
        if not values:
            return {}
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": values[0],
            "p05": values[round(0.05 * (len(values) - 1))],
            "p95": values[round(0.95 * (len(values) - 1))],
            "max": values[-1],
        }

    return {
        "steps": len(rows),
        "steps_with_positive_prediction_energy_r2": sum(
            row["prediction_energy_r2"] > 0 for row in rows
        ),
        "prediction_energy_r2": distribution("prediction_energy_r2"),
        "prediction_cosine_similarity": distribution(
            "prediction_cosine_similarity"
        ),
        "reconstruction_energy_r2": distribution(
            "reconstruction_energy_r2"
        ),
        "encoded_bits_per_gradient_element": distribution(
            "encoded_bits_per_gradient_element"
        ),
        "encode_seconds": distribution("encode_seconds"),
        "compression_latency_ms": distribution("compression_latency_ms"),
        "compression_ratio": distribution("compression_ratio"),
        "compression_throughput_mib_per_second": distribution(
            "compression_throughput_mib_per_second"
        ),
        "allocator_raw_relative_squared_error": distribution(
            "allocator_raw_relative_squared_error"
        ),
        "optimizer_proxy_relative_squared_error": distribution(
            "optimizer_proxy_relative_squared_error"
        ),
        "totals": {
            field: sum(int(row[field]) for row in rows)
            for field in (
                "encoded_bytes",
                "int8_blocks",
                "float16_blocks",
                "float32_blocks",
                "int8_elements",
                "float16_elements",
                "float32_elements",
                "previous_prediction_blocks",
                "zero_prediction_blocks",
                "previous_prediction_elements",
                "zero_prediction_elements",
                "int8_outlier_blocks",
                "int8_outlier_elements",
                "outlier_values",
                "rank1_prediction_tensors",
                "rank1_prediction_blocks",
                "rank1_prediction_elements",
                "rank1_factor_bytes",
                "rht_int8_blocks",
                "rht_int8_elements",
            )
            if all(field in row for row in rows)
        },
    }


def refresh_online_summary(summary_path: Path, metrics_path: Path) -> None:
    """Attach aggregate prediction statistics to an existing online summary."""

    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["prediction_metrics"] = summarize_prediction_metrics(metrics_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_exact(stream, count: int, context: str) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        raise ValueError(f"truncated {context}")
    return value


def _read_header(stream) -> Mapping[str, object]:
    magic, version, size = _FILE_HEADER.unpack(_read_exact(stream, _FILE_HEADER.size, "header"))
    if magic != _MAGIC or version not in {1, 2, 3, 4, 5}:
        raise ValueError("not a supported OTS-DeltaQ artifact")
    return json.loads(_read_exact(stream, size, "header metadata"))
