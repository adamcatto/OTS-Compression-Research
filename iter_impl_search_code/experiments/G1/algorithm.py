"""G1: bounded late-step causal structural rate-frontier experiment.

The encoder enumerates causal low-rank modes derived from the exact preceding
gradient, explicit quantized basis innovations, and sparse/dense residual
codes.  A multiple-choice knapsack chooses one real byte string per tensor
under one global rate budget.  This is deliberately an optimistic post-hoc
screen, not an online training result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import torch

from iter_impl_search_code.experiments.G0.algorithm import _energy, _read_selected_steps


_STREAM_HEADER = struct.Struct("<8sII")
_STREAM_MAGIC = b"OTSG1\0\0\0"
_RANKS = (4, 8, 16, 32, 64)
_CORE_BITS = (4, 8)
_BULK_BITS = (1, 2, 4)
_SPARSE_FRACTIONS = (0.0005, 0.002, 0.01)
_KEYFRAME_INTERVAL = 100
_KNAPSACK_QUANTUM = 16


@dataclass(frozen=True)
class Candidate:
    label: str
    blob: bytes
    squared_error: float
    keyframe_charge_bytes: int

    @property
    def stream_bytes(self) -> int:
        return 4 + len(self.blob)


@dataclass
class BaseEncoding:
    label: str
    metadata: dict[str, object]
    payload: bytes
    reconstruction: torch.Tensor
    keyframe_charge_bytes: int


def _pack_codes(codes: np.ndarray, bits: int) -> bytes:
    values = np.asarray(codes, dtype=np.uint16).reshape(-1)
    if bits == 16:
        return values.astype("<u2", copy=False).tobytes()
    if bits == 8:
        return values.astype(np.uint8, copy=False).tobytes()
    per_byte = 8 // bits
    padding = (-values.size) % per_byte
    if padding:
        values = np.pad(values, (0, padding))
    grouped = values.reshape(-1, per_byte)
    packed = np.zeros(grouped.shape[0], dtype=np.uint8)
    for position in range(per_byte):
        packed |= (grouped[:, position].astype(np.uint8) << (position * bits))
    return packed.tobytes()


def _unpack_codes(payload: bytes, bits: int, count: int) -> np.ndarray:
    if bits == 16:
        return np.frombuffer(payload, dtype="<u2", count=count).astype(np.float32)
    raw = np.frombuffer(payload, dtype=np.uint8)
    if bits == 8:
        return raw[:count].astype(np.float32)
    mask = (1 << bits) - 1
    per_byte = 8 // bits
    output = np.empty(raw.size * per_byte, dtype=np.float32)
    for position in range(per_byte):
        output[position::per_byte] = (raw >> (position * bits)) & mask
    return output[:count]


def _encode_array(value: torch.Tensor, bits: int) -> tuple[bytes, dict[str, object], torch.Tensor]:
    flat = value.detach().to(torch.float32).reshape(-1)
    low = float(torch.min(flat)) if flat.numel() else 0.0
    high = float(torch.max(flat)) if flat.numel() else 0.0
    levels = (1 << bits) - 1
    step = (high - low) / levels if high > low else 0.0
    if step == 0.0:
        codes = torch.zeros_like(flat, dtype=torch.int64)
        decoded = torch.full_like(flat, low)
    else:
        codes = torch.clamp(torch.round((flat - low) / step), 0, levels).to(torch.int64)
        decoded = low + codes.to(torch.float32) * step
    packed = _pack_codes(codes.cpu().numpy(), bits)
    compressed = zlib.compress(packed, level=6)
    use_compressed = len(compressed) < len(packed)
    payload = compressed if use_compressed else packed
    metadata = {
        "bits": bits,
        "compressed": use_compressed,
        "count": flat.numel(),
        "high": high,
        "length": len(payload),
        "low": low,
        "shape": list(value.shape),
    }
    return payload, metadata, decoded.reshape_as(value)


def _decode_array(payload: bytes, metadata: Mapping[str, object]) -> torch.Tensor:
    raw = zlib.decompress(payload) if metadata["compressed"] else payload
    codes = _unpack_codes(raw, int(metadata["bits"]), int(metadata["count"]))
    low = float(metadata["low"])
    high = float(metadata["high"])
    levels = (1 << int(metadata["bits"])) - 1
    step = (high - low) / levels if high > low else 0.0
    decoded = low + torch.from_numpy(codes.copy()) * step
    return decoded.reshape(tuple(int(value) for value in metadata["shape"]))


def _encode_varints(indices: np.ndarray) -> bytes:
    previous = 0
    output = bytearray()
    for position, index in enumerate(indices.tolist()):
        delta = int(index) if position == 0 else int(index) - previous
        previous = int(index)
        while delta >= 0x80:
            output.append((delta & 0x7F) | 0x80)
            delta >>= 7
        output.append(delta)
    return bytes(output)


def _decode_varints(payload: bytes, count: int) -> np.ndarray:
    result = np.empty(count, dtype=np.int64)
    offset = 0
    previous = 0
    for position in range(count):
        value = 0
        shift = 0
        while True:
            byte = payload[offset]
            offset += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        previous = value if position == 0 else previous + value
        result[position] = previous
    return result


def _append_section(metadata: dict[str, object], payload: bytearray, name: str,
                    value: torch.Tensor, bits: int) -> torch.Tensor:
    encoded, description, decoded = _encode_array(value, bits)
    description["offset"] = len(payload)
    metadata.setdefault("sections", {})[name] = description
    payload.extend(encoded)
    return decoded


def _section(blob_payload: bytes, metadata: Mapping[str, object], name: str) -> torch.Tensor:
    description = metadata["sections"][name]
    offset = int(description["offset"])
    length = int(description["length"])
    return _decode_array(blob_payload[offset : offset + length], description)


def _make_blob(base: BaseEncoding, residual_metadata: Optional[dict[str, object]] = None,
               residual_payload: bytes = b"") -> bytes:
    metadata = dict(base.metadata)
    metadata["keyframe_charge_bytes"] = base.keyframe_charge_bytes
    if residual_metadata is not None:
        metadata["residual"] = residual_metadata
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (
        struct.pack("<I", len(metadata_bytes))
        + metadata_bytes
        + base.payload
        + residual_payload
        + bytes(base.keyframe_charge_bytes)
    )


def _basis_charge(rows: int, columns: int, rank: int, mode: str) -> int:
    factors = rows * rank if mode == "left" else columns * rank if mode == "right" else (rows + columns) * rank
    return math.ceil(2 * factors / _KEYFRAME_INTERVAL)


def _svd(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, singular, vh = torch.linalg.svd(value.to(torch.float32), full_matrices=False)
    return u, singular, vh.T


def _base_encodings(current: torch.Tensor, previous: torch.Tensor, tensor_id: int,
                    *, include_oracle_svd: bool) -> list[BaseEncoding]:
    rows, columns = current.shape
    prior_u, _, prior_v = _svd(previous)
    current_u, singular, current_v = _svd(current)
    bases: list[BaseEncoding] = []

    zero_meta = {"tensor_id": tensor_id, "mode": "zero", "sections": {}}
    bases.append(BaseEncoding("zero", zero_meta, b"", torch.zeros_like(current), 0))

    for rank in _RANKS:
        if rank > min(rows, columns):
            continue
        left = prior_u[:, :rank]
        right = prior_v[:, :rank]
        for mode in ("left", "right", "two"):
            coefficient = left.T @ current if mode == "left" else current @ right if mode == "right" else left.T @ current @ right
            for bits in _CORE_BITS:
                metadata: dict[str, object] = {
                    "tensor_id": tensor_id,
                    "mode": f"causal_{mode}",
                    "rank": rank,
                    "sections": {},
                }
                payload = bytearray()
                decoded = _append_section(metadata, payload, "coefficient", coefficient, bits)
                reconstruction = left @ decoded if mode == "left" else decoded @ right.T if mode == "right" else left @ decoded @ right.T
                charge = _basis_charge(rows, columns, rank, mode)
                bases.append(BaseEncoding(f"causal_{mode}_r{rank}_q{bits}", metadata, bytes(payload), reconstruction, charge))

        # Explicitly transmit a coarse innovation from the prior basis.
        aligned_u = current_u[:, :rank].clone()
        aligned_v = current_v[:, :rank].clone()
        signs_u = torch.sign(torch.sum(aligned_u * left, dim=0)); signs_u[signs_u == 0] = 1
        signs_v = torch.sign(torch.sum(aligned_v * right, dim=0)); signs_v[signs_v == 0] = 1
        aligned_u *= signs_u
        aligned_v *= signs_v
        metadata = {"tensor_id": tensor_id, "mode": "basis_innovation", "rank": rank, "sections": {}}
        payload = bytearray()
        delta_u = _append_section(metadata, payload, "delta_u", aligned_u - left, 4)
        delta_v = _append_section(metadata, payload, "delta_v", aligned_v - right, 4)
        decoded_u = torch.linalg.qr(left + delta_u, mode="reduced").Q
        decoded_v = torch.linalg.qr(right + delta_v, mode="reduced").Q
        core = decoded_u.T @ current @ decoded_v
        decoded_core = _append_section(metadata, payload, "coefficient", core, 8)
        reconstruction = decoded_u @ decoded_core @ decoded_v.T
        charge = _basis_charge(rows, columns, rank, "two")
        bases.append(BaseEncoding(f"basis_innovation_r{rank}_q4", metadata, bytes(payload), reconstruction, charge))

    if include_oracle_svd:
        for rank in _RANKS:
            if rank > min(rows, columns):
                continue
            left_scaled = current_u[:, :rank] * singular[:rank]
            for bits in _CORE_BITS:
                candidate_metadata = {"tensor_id": tensor_id, "mode": "oracle_svd", "rank": rank, "sections": {}}
                candidate_payload = bytearray()
                decoded_left = _append_section(candidate_metadata, candidate_payload, "left_scaled", left_scaled, bits)
                decoded_right = _append_section(candidate_metadata, candidate_payload, "right", current_v[:, :rank], bits)
                bases.append(BaseEncoding(f"oracle_svd_r{rank}_q{bits}", candidate_metadata, bytes(candidate_payload), decoded_left @ decoded_right.T, 0))
    return bases


def _residual_variants(base: BaseEncoding, current: torch.Tensor) -> list[Candidate]:
    residual = current - base.reconstruction
    candidates = [Candidate(base.label, _make_blob(base), _energy(residual), base.keyframe_charge_bytes)]

    for bits in _BULK_BITS:
        metadata: dict[str, object] = {"kind": "bulk", "sections": {}}
        payload = bytearray()
        decoded = _append_section(metadata, payload, "bulk", residual, bits)
        combined_base = BaseEncoding(base.label, base.metadata, base.payload, base.reconstruction, base.keyframe_charge_bytes)
        blob = _make_blob(combined_base, metadata, bytes(payload))
        candidates.append(Candidate(f"{base.label}+bulk{bits}", blob, _energy(residual - decoded), base.keyframe_charge_bytes))

    flat = residual.reshape(-1)
    for fraction in _SPARSE_FRACTIONS:
        count = max(1, math.ceil(fraction * flat.numel()))
        indices = torch.topk(torch.abs(flat), count, sorted=False).indices
        indices = torch.sort(indices).values
        values = flat[indices]
        index_raw = _encode_varints(indices.cpu().numpy())
        index_encoded = zlib.compress(index_raw, level=6)
        value_encoded, value_meta, decoded_values = _encode_array(values, 8)
        residual_meta = {
            "count": count,
            "fraction": fraction,
            "index_compressed": True,
            "index_length": len(index_encoded),
            "kind": "sparse",
            "value": value_meta,
        }
        residual_payload = index_encoded + value_encoded
        blob = _make_blob(base, residual_meta, residual_payload)
        reconstructed = torch.zeros_like(flat)
        reconstructed[indices] = decoded_values
        candidates.append(Candidate(f"{base.label}+sparse{fraction:g}", blob, _energy(flat - reconstructed), base.keyframe_charge_bytes))

        if fraction in (0.002, 0.01):
            bulk_source = flat.clone()
            bulk_source[indices] = 0.0
            bulk_encoded, bulk_meta, decoded_bulk = _encode_array(bulk_source, 4)
            decoded_bulk[indices] = 0.0
            residual_meta = {
                "bulk": bulk_meta,
                "count": count,
                "fraction": fraction,
                "index_compressed": True,
                "index_length": len(index_encoded),
                "kind": "sparse_bulk",
                "value": value_meta,
            }
            residual_payload = index_encoded + value_encoded + bulk_encoded
            blob = _make_blob(base, residual_meta, residual_payload)
            reconstructed = decoded_bulk
            reconstructed[indices] = decoded_values
            candidates.append(Candidate(f"{base.label}+sparse{fraction:g}+bulk4", blob, _energy(flat - reconstructed), base.keyframe_charge_bytes))
    return candidates


def _pareto(candidates: Iterable[Candidate], limit: int = 48) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda candidate: (candidate.stream_bytes, candidate.squared_error))
    frontier: list[Candidate] = []
    best_error = math.inf
    for candidate in ordered:
        if candidate.squared_error < best_error * (1.0 - 1e-12):
            frontier.append(candidate)
            best_error = candidate.squared_error
    if len(frontier) <= limit:
        return frontier
    indices = np.unique(np.linspace(0, len(frontier) - 1, limit).round().astype(int))
    return [frontier[index] for index in indices]


def _tensor_candidates(current: torch.Tensor, previous: Optional[torch.Tensor], tensor_id: int,
                       *, include_oracle_svd: bool) -> list[Candidate]:
    if current.ndim == 3 and current.shape[0] == 1:
        matrix = current[0].to(torch.float32)
        prior_matrix = None if previous is None else previous[0].to(torch.float32)
    elif current.ndim == 2:
        matrix = current.to(torch.float32)
        prior_matrix = None if previous is None else previous.to(torch.float32)
    else:
        matrix = current.reshape(1, -1).to(torch.float32)
        prior_matrix = None if previous is None else previous.reshape(1, -1).to(torch.float32)

    bases: list[BaseEncoding]
    if prior_matrix is not None and matrix.shape == prior_matrix.shape and min(matrix.shape) >= 4:
        bases = _base_encodings(matrix, prior_matrix, tensor_id, include_oracle_svd=include_oracle_svd)
    else:
        bases = [BaseEncoding("zero", {"tensor_id": tensor_id, "mode": "zero", "sections": {}}, b"", torch.zeros_like(matrix), 0)]

    # Residual coding is expensive to enumerate. Keep the most useful base
    # frontier, then expand only eight representatives.
    base_scored = []
    for base in bases:
        blob = _make_blob(base)
        base_scored.append(Candidate(base.label, blob, _energy(matrix - base.reconstruction), base.keyframe_charge_bytes))
    selected_labels = {candidate.label for candidate in _pareto(base_scored, limit=8)}
    candidates: list[Candidate] = []
    for base in bases:
        if base.label in selected_labels:
            candidates.extend(_residual_variants(base, matrix))
        else:
            residual = matrix - base.reconstruction
            candidates.append(Candidate(base.label, _make_blob(base), _energy(residual), base.keyframe_charge_bytes))
    return _pareto(candidates)


def _select(candidates: list[list[Candidate]], budget_bytes: int) -> list[Candidate]:
    usable = budget_bytes - _STREAM_HEADER.size
    capacity = usable // _KNAPSACK_QUANTUM
    dp = np.full(capacity + 1, -np.inf, dtype=np.float64)
    dp[0] = 0.0
    histories: list[tuple[np.ndarray, np.ndarray]] = []
    for choices in candidates:
        new = np.full_like(dp, -np.inf)
        chosen = np.full(capacity + 1, -1, dtype=np.int16)
        previous_budget = np.full(capacity + 1, -1, dtype=np.int32)
        for index, candidate in enumerate(choices):
            units = math.ceil(candidate.stream_bytes / _KNAPSACK_QUANTUM)
            if units > capacity:
                continue
            source = dp[: capacity + 1 - units]
            gain = -candidate.squared_error
            target = source + gain
            destination = new[units:]
            mask = target > destination
            destination[mask] = target[mask]
            positions = np.nonzero(mask)[0] + units
            chosen[positions] = index
            previous_budget[positions] = np.nonzero(mask)[0]
        if not np.isfinite(new).any():
            raise ValueError("global budget cannot hold one code per tensor")
        dp = new
        histories.append((chosen, previous_budget))
    budget = int(np.nanargmax(dp))
    selected: list[Candidate] = []
    for tensor_index in range(len(candidates) - 1, -1, -1):
        chosen, previous_budget = histories[tensor_index]
        candidate_index = int(chosen[budget])
        selected.append(candidates[tensor_index][candidate_index])
        budget = int(previous_budget[budget])
    selected.reverse()
    return selected


def _assemble(selected: list[Candidate]) -> bytes:
    output = bytearray(_STREAM_HEADER.pack(_STREAM_MAGIC, 1, len(selected)))
    for candidate in selected:
        output.extend(struct.pack("<I", len(candidate.blob)))
        output.extend(candidate.blob)
    return bytes(output)


def _split_blob(blob: bytes) -> tuple[dict[str, object], bytes]:
    metadata_size = struct.unpack_from("<I", blob, 0)[0]
    metadata = json.loads(blob[4 : 4 + metadata_size])
    charge = int(metadata.get("keyframe_charge_bytes", 0))
    return metadata, blob[4 + metadata_size : len(blob) - charge if charge else len(blob)]


def _decode_blob(blob: bytes, previous: Optional[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    metadata, payload = _split_blob(blob)
    sections = metadata.get("sections", {})
    base_length = max((int(item["offset"]) + int(item["length"]) for item in sections.values()), default=0)
    base_payload = payload[:base_length]
    mode = metadata["mode"]
    matrix_shape = shape[1:] if len(shape) == 3 and shape[0] == 1 else (shape if len(shape) == 2 else (1, math.prod(shape)))
    if mode == "zero":
        reconstruction = torch.zeros(matrix_shape, dtype=torch.float32)
    elif str(mode).startswith("causal_"):
        if previous is None:
            raise ValueError("causal chunk requires prior tensor")
        prior = previous[0] if previous.ndim == 3 and previous.shape[0] == 1 else previous.reshape(matrix_shape)
        prior_u, _, prior_v = _svd(prior)
        rank = int(metadata["rank"])
        coefficient = _section(base_payload, metadata, "coefficient")
        left, right = prior_u[:, :rank], prior_v[:, :rank]
        family = str(mode).removeprefix("causal_")
        reconstruction = left @ coefficient if family == "left" else coefficient @ right.T if family == "right" else left @ coefficient @ right.T
    elif mode == "basis_innovation":
        if previous is None:
            raise ValueError("basis innovation requires prior tensor")
        prior = previous[0] if previous.ndim == 3 and previous.shape[0] == 1 else previous.reshape(matrix_shape)
        prior_u, _, prior_v = _svd(prior)
        rank = int(metadata["rank"])
        left = torch.linalg.qr(prior_u[:, :rank] + _section(base_payload, metadata, "delta_u"), mode="reduced").Q
        right = torch.linalg.qr(prior_v[:, :rank] + _section(base_payload, metadata, "delta_v"), mode="reduced").Q
        reconstruction = left @ _section(base_payload, metadata, "coefficient") @ right.T
    elif mode == "oracle_svd":
        reconstruction = _section(base_payload, metadata, "left_scaled") @ _section(base_payload, metadata, "right").T
    else:
        raise ValueError(f"unknown mode: {mode}")

    residual = metadata.get("residual")
    if residual is not None:
        residual_payload = payload[base_length:]
        if residual["kind"] == "bulk":
            reconstruction = reconstruction + _section(residual_payload, residual, "bulk")
        elif residual["kind"] == "sparse":
            index_length = int(residual["index_length"])
            index_raw = zlib.decompress(residual_payload[:index_length])
            indices = _decode_varints(index_raw, int(residual["count"]))
            value_meta = residual["value"]
            values = _decode_array(residual_payload[index_length : index_length + int(value_meta["length"])], value_meta)
            flat = reconstruction.reshape(-1)
            flat[torch.from_numpy(indices)] += values.reshape(-1)
        elif residual["kind"] == "sparse_bulk":
            index_length = int(residual["index_length"])
            index_raw = zlib.decompress(residual_payload[:index_length])
            indices = _decode_varints(index_raw, int(residual["count"]))
            value_meta = residual["value"]
            value_start = index_length
            value_end = value_start + int(value_meta["length"])
            values = _decode_array(residual_payload[value_start:value_end], value_meta)
            bulk_meta = residual["bulk"]
            bulk = _decode_array(residual_payload[value_end:value_end + int(bulk_meta["length"])], bulk_meta).reshape(-1)
            tensor_indices = torch.from_numpy(indices)
            bulk[tensor_indices] = 0.0
            flat = reconstruction.reshape(-1)
            flat += bulk
            flat[tensor_indices] += values.reshape(-1)
        else:
            raise ValueError(f"unknown residual mode: {residual['kind']}")
    return reconstruction.reshape(shape)


def decode_stream(stream: bytes, previous: Mapping[str, torch.Tensor],
                  catalog: list[tuple[str, tuple[int, ...]]]) -> dict[str, torch.Tensor]:
    magic, version, count = _STREAM_HEADER.unpack_from(stream, 0)
    if magic != _STREAM_MAGIC or version != 1 or count != len(catalog):
        raise ValueError("invalid G1 stream header")
    offset = _STREAM_HEADER.size
    output: dict[str, torch.Tensor] = {}
    for expected_id, (name, shape) in enumerate(catalog):
        length = struct.unpack_from("<I", stream, offset)[0]
        offset += 4
        blob = stream[offset : offset + length]
        offset += length
        metadata, _ = _split_blob(blob)
        if int(metadata["tensor_id"]) != expected_id:
            raise ValueError("tensor order mismatch")
        output[name] = _decode_blob(blob, previous.get(name), shape)
    if offset != len(stream):
        raise ValueError("trailing bytes in G1 stream")
    return output


def _cosine(original: Mapping[str, torch.Tensor], decoded: Mapping[str, torch.Tensor]) -> float:
    dot = sum(float(torch.sum(original[name].to(torch.float64) * decoded[name].to(torch.float64))) for name in original)
    left = sum(_energy(value) for value in original.values())
    right = sum(_energy(value) for value in decoded.values())
    return dot / math.sqrt(left * right) if left > 0 and right > 0 else 0.0


def run_step(current: Mapping[str, torch.Tensor], previous: Mapping[str, torch.Tensor],
             *, budget_bits_per_value: float, include_oracle_svd: bool = False) -> tuple[dict[str, object], bytes]:
    names = sorted(current)
    total_values = sum(value.numel() for value in current.values())
    uncompressed_bytes = total_values * 4
    budget_bytes = math.floor(total_values * budget_bits_per_value / 8)
    encode_started = time.perf_counter()
    all_candidates = [
        _tensor_candidates(current[name], previous.get(name), tensor_id, include_oracle_svd=include_oracle_svd)
        for tensor_id, name in enumerate(names)
    ]
    selected = _select(all_candidates, budget_bytes)
    stream = _assemble(selected)
    encode_seconds = time.perf_counter() - encode_started
    if len(stream) > budget_bytes:
        raise AssertionError("knapsack emitted an over-budget stream")
    decode_started = time.perf_counter()
    decoded = decode_stream(stream, previous, [(name, tuple(current[name].shape)) for name in names])
    decode_seconds = time.perf_counter() - decode_started
    total_energy = sum(_energy(value) for value in current.values())
    squared_error = sum(_energy(current[name] - decoded[name]) for name in names)
    modes: dict[str, int] = {}
    for candidate in selected:
        modes[candidate.label] = modes.get(candidate.label, 0) + 1
    row: dict[str, object] = {
        "uncompressed_bytes": uncompressed_bytes,
        "compressed_bytes": len(stream),
        "budget_bytes": budget_bytes,
        "compression_ratio": uncompressed_bytes / len(stream),
        "bits_per_value": 8 * len(stream) / total_values,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "encode_mvalues_per_second": total_values / encode_seconds / 1e6,
        "decode_mvalues_per_second": total_values / decode_seconds / 1e6,
        "gradient_energy_r2": 1.0 - squared_error / total_energy,
        "cosine_similarity": _cosine(current, decoded),
        "keyframe_charge_bytes": sum(candidate.keyframe_charge_bytes for candidate in selected),
        "tensor_count": len(names),
        "selected_modes": json.dumps(modes, sort_keys=True),
        "candidate_kind": "oracle_svd_augmented" if include_oracle_svd else "causal",
    }
    return row, stream


def run_trace(trace: Path, steps: list[int], *, budget_bits_per_value: float = 0.32,
              fidelity_r2: float = 0.99,
              abort_r2: float = 0.90) -> tuple[list[dict[str, object]], dict[str, object]]:
    requested = set(steps)
    requested.update(step - 1 for step in steps)
    snapshots = _read_selected_steps(trace, requested)
    rows: list[dict[str, object]] = []
    stopped_early = False
    for step in steps:
        causal, _ = run_step(snapshots[step], snapshots[step - 1], budget_bits_per_value=budget_bits_per_value)
        frontier, _ = run_step(
            snapshots[step],
            snapshots[step - 1],
            budget_bits_per_value=budget_bits_per_value,
            include_oracle_svd=True,
        )
        row: dict[str, object] = {"step": step}
        row.update({f"causal_{key}": value for key, value in causal.items() if key != "candidate_kind"})
        row.update({f"frontier_{key}": value for key, value in frontier.items() if key != "candidate_kind"})
        rows.append(row)
        if float(frontier["gradient_energy_r2"]) < abort_r2:
            stopped_early = True
            break
    fidelity_failed = any(
        float(row["frontier_gradient_energy_r2"]) < fidelity_r2 for row in rows
    )
    if stopped_early:
        decision = "reject_gradient_only_g1_aborted_below_continuation_cutoff"
    elif fidelity_failed:
        decision = "reject_gradient_only_g1_completed_sample"
    else:
        decision = "authorize_short_causal_prefix"
    summary = {
        "trace": str(trace),
        "requested_steps": steps,
        "executed_steps": [int(row["step"]) for row in rows],
        "hard_target_bits_per_value": budget_bits_per_value,
        "fidelity_gate_r2": fidelity_r2,
        "continuation_abort_r2": abort_r2,
        "stopped_early": stopped_early,
        "completed_requested_steps": len(rows) == len(steps),
        "decision": decision,
        "gate_frontier": "paid_current_svd_augmented_selector",
        "rows": rows,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--steps", default="100,500,999")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--budget-bpv", type=float, default=0.32)
    parser.add_argument("--fidelity-r2", type=float, default=0.99)
    parser.add_argument("--abort-r2", type=float, default=0.90)
    args = parser.parse_args()
    rows, summary = run_trace(
        args.trace,
        [int(value) for value in args.steps.split(",")],
        budget_bits_per_value=args.budget_bpv,
        fidelity_r2=args.fidelity_r2,
        abort_r2=args.abort_r2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
