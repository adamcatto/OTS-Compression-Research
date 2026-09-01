"""Lightweight structural-feasibility audit for the exact gradient trace.

This is an analysis, not a compressor.  It indexes the append-only trace and
decodes only explicitly requested steps (plus their immediate predecessors),
so it does not create another multi-gigabyte artifact or replay a completed
compression experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import torch


_FILE_HEADER = struct.Struct("<8sHI")
_RECORD_HEADER = struct.Struct("<4sQIQI")
_FILE_MAGIC = b"OTSGRAD1"
_RECORD_MAGIC = b"STEP"
_LAYER = re.compile(r"^(.*\.layers\.)(\d+)(\..*)$")
_RANKS = (1, 2, 4, 8, 16, 32, 64)
_TT_RANKS = (1, 2, 4, 8, 16)
_FFT_FRACTIONS = (0.001, 0.005, 0.01, 0.05)


@dataclass(frozen=True)
class _RecordIndex:
    step: int
    metadata: bytes
    payload_offset: int
    payload_size: int
    checksum: int


def _index_trace(path: Path) -> list[_RecordIndex]:
    records: list[_RecordIndex] = []
    with path.open("rb") as stream:
        fixed = stream.read(_FILE_HEADER.size)
        if len(fixed) != _FILE_HEADER.size:
            raise ValueError("truncated trace header")
        magic, version, metadata_size = _FILE_HEADER.unpack(fixed)
        if magic != _FILE_MAGIC or version != 1:
            raise ValueError("unsupported trace format")
        stream.seek(metadata_size, 1)
        while True:
            fixed = stream.read(_RECORD_HEADER.size)
            if not fixed:
                break
            if len(fixed) != _RECORD_HEADER.size:
                raise ValueError("truncated record header")
            magic, step, record_metadata_size, payload_size, checksum = (
                _RECORD_HEADER.unpack(fixed)
            )
            if magic != _RECORD_MAGIC:
                raise ValueError(f"invalid record marker at step {step}")
            metadata = stream.read(record_metadata_size)
            if len(metadata) != record_metadata_size:
                raise ValueError(f"truncated metadata at step {step}")
            payload_offset = stream.tell()
            records.append(
                _RecordIndex(
                    step=step,
                    metadata=metadata,
                    payload_offset=payload_offset,
                    payload_size=payload_size,
                    checksum=checksum,
                )
            )
            stream.seek(payload_size, 1)
    return records


def _read_selected_steps(
    path: Path, requested_steps: Iterable[int]
) -> dict[int, dict[str, torch.Tensor]]:
    wanted = set(requested_steps)
    index = {record.step: record for record in _index_trace(path)}
    missing = sorted(wanted.difference(index))
    if missing:
        raise ValueError(f"steps are absent from trace: {missing}")
    result: dict[int, dict[str, torch.Tensor]] = {}
    with path.open("rb") as stream:
        for step in sorted(wanted):
            record = index[step]
            stream.seek(record.payload_offset)
            payload = stream.read(record.payload_size)
            if len(payload) != record.payload_size:
                raise ValueError(f"truncated payload at step {step}")
            if zlib.crc32(payload, zlib.crc32(record.metadata)) != record.checksum:
                raise ValueError(f"checksum mismatch at step {step}")
            descriptors = json.loads(record.metadata).get("tensors", [])
            tensors: dict[str, torch.Tensor] = {}
            for descriptor in descriptors:
                if not descriptor.get("present", False):
                    continue
                offset = int(descriptor["offset"])
                nbytes = int(descriptor["nbytes"])
                raw = payload[offset : offset + nbytes]
                dtype = getattr(torch, descriptor["dtype"])
                value = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                tensors[descriptor["name"]] = (
                    value.view(dtype).reshape(descriptor["shape"]).clone()
                )
            result[step] = tensors
    return result


def _matrix_view(value: torch.Tensor) -> Optional[torch.Tensor]:
    if value.ndim == 2:
        return value.to(torch.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        return value[0].to(torch.float32)
    return None


def _energy(value: torch.Tensor) -> float:
    return float(torch.sum(value.to(torch.float64) ** 2))


def _fraction(numerator: float, denominator: float) -> Optional[float]:
    return None if denominator <= 0 else min(1.0, max(0.0, numerator / denominator))


def _balanced_factors(size: int) -> Optional[tuple[int, int]]:
    for divisor in range(math.isqrt(size), 1, -1):
        if size % divisor == 0:
            return divisor, size // divisor
    return None


def kronecker_rearrangement(matrix: torch.Tensor) -> Optional[torch.Tensor]:
    """Return the Van Loan rearrangement used for best Kronecker rank."""

    rows = _balanced_factors(matrix.shape[0])
    columns = _balanced_factors(matrix.shape[1])
    if rows is None or columns is None:
        return None
    a, b = rows
    c, d = columns
    return matrix.reshape(a, b, c, d).permute(0, 2, 1, 3).reshape(a * c, b * d)


def _architecture_shape(name: str, matrix: torch.Tensor) -> tuple[int, ...]:
    """Expose known transformer axes without changing element order."""

    rows, columns = matrix.shape
    if "self_attn.in_proj_weight" in name and (rows, columns) == (768, 256):
        return (3, 8, 32, 8, 32)
    if "self_attn.out_proj.weight" in name and (rows, columns) == (256, 256):
        return (8, 32, 8, 32)
    if "linear1.weight" in name and (rows, columns) == (1024, 256):
        return (4, 8, 32, 8, 32)
    if "linear2.weight" in name and (rows, columns) == (256, 1024):
        return (8, 32, 4, 8, 32)
    if name == "token_embedding.weight" and columns == 256:
        return (rows, 8, 32)
    if name == "position_embedding" and (rows, columns) == (256, 256):
        return (rows, 8, 32)
    return (rows, columns)


def _tensor_train_approximation(
    matrix: torch.Tensor, shape: tuple[int, ...], maximum_rank: int
) -> tuple[torch.Tensor, int]:
    """Return the deterministic TT-SVD approximation and parameter count."""

    if math.prod(shape) != matrix.numel():
        raise ValueError("tensorization shape does not match matrix")
    working = matrix.reshape(shape)
    cores: list[torch.Tensor] = []
    previous_rank = 1
    parameters = 0
    for axis, extent in enumerate(shape[:-1]):
        unfolding = working.reshape(previous_rank * extent, -1)
        left, singular, right = torch.linalg.svd(unfolding, full_matrices=False)
        rank = min(maximum_rank, singular.numel())
        core = left[:, :rank].reshape(previous_rank, extent, rank)
        cores.append(core)
        parameters += core.numel()
        working = singular[:rank, None] * right[:rank]
        previous_rank = rank
    final = working.reshape(previous_rank, shape[-1], 1)
    cores.append(final)
    parameters += final.numel()

    reconstructed = cores[0]
    for core in cores[1:]:
        reconstructed = torch.tensordot(reconstructed, core, dims=([-1], [0]))
    return reconstructed.squeeze(0).squeeze(-1).reshape_as(matrix), parameters


def gaussian_residual_rate_lower_bound(
    residual_energy_fraction: Optional[float], *, distortion_fraction: float = 0.01
) -> Optional[float]:
    """Ideal iid-Gaussian residual rate in bits per original value."""

    if residual_energy_fraction is None:
        return None
    if residual_energy_fraction <= distortion_fraction:
        return 0.0
    return 0.5 * math.log2(residual_energy_fraction / distortion_fraction)


def _spectrum_energy_fractions(
    singular_values: torch.Tensor, ranks: Iterable[int]
) -> dict[int, float]:
    squared = singular_values.to(torch.float64) ** 2
    total = float(torch.sum(squared))
    cumulative = torch.cumsum(squared, dim=0)
    return {
        rank: float(cumulative[min(rank, len(cumulative)) - 1]) / total
        for rank in ranks
        if total > 0 and rank <= len(cumulative)
    }


def _fft_energy_fractions(matrix: torch.Tensor) -> tuple[dict[float, float], dict[float, float]]:
    spectrum = torch.fft.fft2(matrix, norm="ortho")
    power = torch.abs(spectrum).to(torch.float64).reshape(-1) ** 2
    total = float(torch.sum(power))
    ordered = torch.sort(power, descending=True).values
    cumulative = torch.cumsum(ordered, dim=0)

    row_frequency = torch.fft.fftfreq(matrix.shape[0]).to(torch.float64)
    column_frequency = torch.fft.fftfreq(matrix.shape[1]).to(torch.float64)
    radius = (
        row_frequency[:, None] ** 2 + column_frequency[None, :] ** 2
    ).reshape(-1)
    radial_order = torch.argsort(radius)
    radial_cumulative = torch.cumsum(power[radial_order], dim=0)

    best: dict[float, float] = {}
    fixed_low: dict[float, float] = {}
    for fraction in _FFT_FRACTIONS:
        count = max(1, math.ceil(fraction * power.numel()))
        best[fraction] = float(cumulative[count - 1]) / total
        fixed_low[fraction] = float(radial_cumulative[count - 1]) / total
    return best, fixed_low


def _projection_fractions(
    current: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    ranks: Iterable[int],
) -> dict[int, tuple[float, float, float]]:
    total = _energy(current)
    values: dict[int, tuple[float, float, float]] = {}
    for rank in ranks:
        if rank > min(left.shape[1], right.shape[1]):
            continue
        u = left[:, :rank]
        v = right[:, :rank]
        left_energy = _energy(u.T @ current)
        right_energy = _energy(current @ v)
        core_energy = _energy(u.T @ current @ v)
        values[rank] = (
            left_energy / total,
            right_energy / total,
            core_energy / total,
        )
    return values


def _shared_bases(
    matrices: Mapping[str, torch.Tensor], ranks: Iterable[int]
) -> dict[str, tuple[torch.Tensor, torch.Tensor, int]]:
    groups: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for name, matrix in matrices.items():
        match = _LAYER.match(name)
        if match is None:
            continue
        group = match.group(1) + "*" + match.group(3)
        groups.setdefault(group, []).append((name, matrix))

    output: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
    maximum_rank = max(ranks)
    for members in groups.values():
        if len(members) < 2 or len({tuple(value.shape) for _, value in members}) != 1:
            continue
        rows, columns = members[0][1].shape
        row_covariance = torch.zeros((rows, rows), dtype=torch.float32)
        column_covariance = torch.zeros((columns, columns), dtype=torch.float32)
        for _, value in members:
            row_covariance.add_(value @ value.T)
            column_covariance.add_(value.T @ value)
        _, left = torch.linalg.eigh(row_covariance)
        _, right = torch.linalg.eigh(column_covariance)
        left = torch.flip(left, dims=(1,))[:, :maximum_rank]
        right = torch.flip(right, dims=(1,))[:, :maximum_rank]
        for name, _ in members:
            output[name] = (left, right, len(members))
    return output


def _empirical_scalar_entropy(
    residual: torch.Tensor,
    *,
    total_energy: float,
    total_values: int,
    target_relative_sse: float = 0.01,
    maximum_samples: int = 65_536,
) -> tuple[float, float]:
    """Optimistic zero-order entropy for a tuned uniform residual quantizer."""

    flat = residual.detach().reshape(-1).to(torch.float64)
    if flat.numel() > maximum_samples:
        indices = torch.linspace(0, flat.numel() - 1, maximum_samples).long()
        flat = flat[indices]
    allowed_mse = target_relative_sse * total_energy / total_values
    residual_mse = float(torch.mean(flat**2))
    if residual_mse <= allowed_mse:
        return 0.0, residual_mse * total_values / total_energy

    step = math.sqrt(12.0 * allowed_mse)
    for _ in range(10):
        quantized = torch.round(flat / step)
        mse = float(torch.mean((flat - step * quantized) ** 2))
        if mse <= 0:
            break
        step *= math.sqrt(allowed_mse / mse)
    quantized = torch.round(flat / step).to(torch.int64)
    _, counts = torch.unique(quantized, return_counts=True)
    probabilities = counts.to(torch.float64) / counts.sum()
    entropy = float(-torch.sum(probabilities * torch.log2(probabilities)))
    achieved = float(torch.mean((flat - step * quantized.to(torch.float64)) ** 2))
    return entropy, achieved * total_values / total_energy


def _nullable(value: Optional[float]) -> str:
    return "" if value is None else repr(float(value))


def audit_trace(trace: Path, steps: list[int]) -> tuple[list[dict[str, object]], dict[str, object]]:
    requested = set(steps)
    requested.update(step - 1 for step in steps if step > 0)
    snapshots = _read_selected_steps(trace, requested)
    rows: list[dict[str, object]] = []

    for step in steps:
        current = {
            name: matrix
            for name, value in snapshots[step].items()
            if (matrix := _matrix_view(value)) is not None
        }
        previous = {
            name: matrix
            for name, value in snapshots.get(step - 1, {}).items()
            if (matrix := _matrix_view(value)) is not None
        }
        shared = _shared_bases(previous, _RANKS)

        for name, matrix in current.items():
            total_energy = _energy(matrix)
            u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
            self_fractions = _spectrum_energy_fractions(singular, _RANKS)
            best_fft, fixed_fft = _fft_energy_fractions(matrix)
            rearranged = kronecker_rearrangement(matrix)
            kron_fractions: dict[int, float] = {}
            if rearranged is not None:
                kron_fractions = _spectrum_energy_fractions(
                    torch.linalg.svdvals(rearranged), _RANKS
                )
            tensor_shape = _architecture_shape(name, matrix)
            tt_metrics: dict[int, tuple[float, int]] = {}
            for rank in _TT_RANKS:
                approximation, parameters = _tensor_train_approximation(
                    matrix, tensor_shape, rank
                )
                tt_metrics[rank] = (_energy(approximation) / total_energy, parameters)

            prior = previous.get(name)
            temporal_raw_r2: Optional[float] = None
            temporal_fitted_r2: Optional[float] = None
            local_projections: dict[int, tuple[float, float, float]] = {}
            causal_residual: Optional[torch.Tensor] = None
            if prior is not None and prior.shape == matrix.shape:
                prior_u, _, prior_vh = torch.linalg.svd(prior, full_matrices=False)
                local_projections = _projection_fractions(
                    matrix, prior_u, prior_vh.T, _RANKS
                )
                residual = matrix - prior
                temporal_raw_r2 = 1.0 - _energy(residual) / total_energy
                coefficient = float(torch.sum(matrix * prior)) / max(_energy(prior), 1e-30)
                temporal_fitted_r2 = 1.0 - _energy(matrix - coefficient * prior) / total_energy
                if 8 in local_projections:
                    causal_residual = matrix - (
                        prior_u[:, :8] @ (prior_u[:, :8].T @ matrix @ prior_vh[:8].T) @ prior_vh[:8]
                    )

            shared_projections: dict[int, tuple[float, float, float]] = {}
            shared_layers = 0
            if name in shared:
                left, right, shared_layers = shared[name]
                shared_projections = _projection_fractions(
                    matrix, left, right, _RANKS
                )

            oracle_residual = matrix - u[:, :8] @ torch.diag(singular[:8]) @ vh[:8]
            oracle_entropy, oracle_quantized_sse = _empirical_scalar_entropy(
                oracle_residual,
                total_energy=total_energy,
                total_values=matrix.numel(),
            )
            causal_entropy: Optional[float] = None
            causal_quantized_sse: Optional[float] = None
            if causal_residual is not None:
                causal_entropy, causal_quantized_sse = _empirical_scalar_entropy(
                    causal_residual,
                    total_energy=total_energy,
                    total_values=matrix.numel(),
                )

            row: dict[str, object] = {
                "step": step,
                "tensor_name": name,
                "rows": matrix.shape[0],
                "columns": matrix.shape[1],
                "values": matrix.numel(),
                "energy": total_energy,
                "temporal_previous_r2": _nullable(temporal_raw_r2),
                "temporal_oracle_scalar_r2": _nullable(temporal_fitted_r2),
                "shared_basis_layers": shared_layers,
                "tensorization_shape": "x".join(str(value) for value in tensor_shape),
                "oracle_rank8_scalar_entropy_bpv": oracle_entropy,
                "oracle_rank8_quantized_relative_sse": oracle_quantized_sse,
                "causal_rank8_scalar_entropy_bpv": _nullable(causal_entropy),
                "causal_rank8_quantized_relative_sse": _nullable(causal_quantized_sse),
            }
            for rank in _RANKS:
                self_fraction = self_fractions.get(rank)
                row[f"self_rank{rank}_energy_fraction"] = _nullable(self_fraction)
                row[f"self_rank{rank}_gaussian_residual_bpv"] = _nullable(
                    gaussian_residual_rate_lower_bound(
                        None if self_fraction is None else 1.0 - self_fraction
                    )
                )
                row[f"self_rank{rank}_fp16_factor_bpv"] = (
                    16.0 * rank * (matrix.shape[0] + matrix.shape[1] + 1) / matrix.numel()
                    if self_fraction is not None
                    else ""
                )
                local = local_projections.get(rank)
                row[f"causal_rank{rank}_left_energy_fraction"] = _nullable(
                    None if local is None else local[0]
                )
                row[f"causal_rank{rank}_right_energy_fraction"] = _nullable(
                    None if local is None else local[1]
                )
                row[f"causal_rank{rank}_two_sided_energy_fraction"] = _nullable(
                    None if local is None else local[2]
                )
                row[f"causal_rank{rank}_gaussian_residual_bpv"] = _nullable(
                    gaussian_residual_rate_lower_bound(
                        None if local is None else 1.0 - local[2]
                    )
                )
                row[f"causal_rank{rank}_fp16_core_bpv"] = (
                    16.0 * rank * rank / matrix.numel() if local is not None else ""
                )
                shared_projection = shared_projections.get(rank)
                row[f"shared_causal_rank{rank}_two_sided_energy_fraction"] = _nullable(
                    None if shared_projection is None else shared_projection[2]
                )
                kron_fraction = kron_fractions.get(rank)
                row[f"kronecker_rank{rank}_energy_fraction"] = _nullable(kron_fraction)
                if rank in tt_metrics:
                    tt_fraction, tt_parameters = tt_metrics[rank]
                    row[f"tt_rank{rank}_energy_fraction"] = tt_fraction
                    row[f"tt_rank{rank}_parameters"] = tt_parameters
                    row[f"tt_rank{rank}_fp16_factor_bpv"] = (
                        16.0 * tt_parameters / matrix.numel()
                    )
            for fraction in _FFT_FRACTIONS:
                label = str(fraction).replace(".", "p")
                row[f"fft_best_{label}_energy_fraction"] = best_fft[fraction]
                row[f"fft_fixed_low_{label}_energy_fraction"] = fixed_fft[fraction]
            rows.append(row)

    value_weight = sum(int(row["values"]) for row in rows)

    def weighted(column: str) -> Optional[float]:
        present = [
            (int(row["values"]), float(row[column]))
            for row in rows
            if row.get(column) not in (None, "")
        ]
        denominator = sum(weight for weight, _ in present)
        return None if denominator == 0 else sum(weight * value for weight, value in present) / denominator

    total_step_values = sum(value.numel() for value in snapshots[steps[0]].values())
    summary: dict[str, object] = {
        "trace": str(trace),
        "trace_steps_indexed": len(_index_trace(trace)),
        "sampled_steps": steps,
        "decoded_steps": sorted(requested),
        "matrix_rows": len(rows),
        "sampled_matrix_values": value_weight,
        "values_per_full_step": total_step_values,
        "hard_target_bits_per_value": 0.32,
        "hard_target_bytes_per_full_step": total_step_values * 0.32 / 8.0,
        "gaussian_signal_fraction_needed_before_residual_coding": 1.0 - 0.01 * 2.0**0.64,
        "weighted_metrics": {
            "temporal_previous_r2": weighted("temporal_previous_r2"),
            "temporal_oracle_scalar_r2": weighted("temporal_oracle_scalar_r2"),
            "self_rank1_energy_fraction": weighted("self_rank1_energy_fraction"),
            "self_rank8_energy_fraction": weighted("self_rank8_energy_fraction"),
            "self_rank32_energy_fraction": weighted("self_rank32_energy_fraction"),
            "causal_rank8_left_energy_fraction": weighted("causal_rank8_left_energy_fraction"),
            "causal_rank8_right_energy_fraction": weighted("causal_rank8_right_energy_fraction"),
            "causal_rank8_two_sided_energy_fraction": weighted("causal_rank8_two_sided_energy_fraction"),
            "shared_causal_rank8_two_sided_energy_fraction": weighted("shared_causal_rank8_two_sided_energy_fraction"),
            "kronecker_rank1_energy_fraction": weighted("kronecker_rank1_energy_fraction"),
            "kronecker_rank8_energy_fraction": weighted("kronecker_rank8_energy_fraction"),
            "tt_rank8_energy_fraction": weighted("tt_rank8_energy_fraction"),
            "tt_rank8_fp16_factor_bpv": weighted("tt_rank8_fp16_factor_bpv"),
            "fft_best_0p01_energy_fraction": weighted("fft_best_0p01_energy_fraction"),
            "fft_fixed_low_0p01_energy_fraction": weighted("fft_fixed_low_0p01_energy_fraction"),
            "oracle_rank8_scalar_entropy_bpv": weighted("oracle_rank8_scalar_entropy_bpv"),
            "causal_rank8_scalar_entropy_bpv": weighted("causal_rank8_scalar_entropy_bpv"),
            "self_rank8_gaussian_residual_bpv": weighted("self_rank8_gaussian_residual_bpv"),
            "causal_rank8_gaussian_residual_bpv": weighted("causal_rank8_gaussian_residual_bpv"),
        },
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--steps", default="0,1,100,500,999")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(",")]
    rows, summary = audit_trace(args.trace, steps)
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
