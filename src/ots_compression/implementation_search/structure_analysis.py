"""Measure decoder-usable structure in an online gradient trace.

The measurements are prediction upper bounds, not compression results.  They
use only the current tensor plus state that an online encoder could already
have: earlier decoded steps, earlier homologous layers in the current step, or
earlier coordinates in the current tensor.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import torch

from ..core.gradient_trace import GradientTraceReader
from ..core.settings import Settings


_LAYER = re.compile(r"^(.*\.layers\.)(\d+)(\..*)$")


@dataclass
class PredictionTotals:
    energy: float = 0.0
    raw_sse: float = 0.0
    fitted_sse: float = 0.0
    count: int = 0
    elements: int = 0
    coefficients: list[float] = field(default_factory=list)

    def add(
        self,
        target: torch.Tensor,
        predictors: Iterable[torch.Tensor],
        *,
        raw_predictor: Optional[torch.Tensor] = None,
    ) -> None:
        y = target.reshape(-1).to(torch.float32)
        xs = [value.reshape(-1).to(torch.float32) for value in predictors]
        if not xs or any(value.shape != y.shape for value in xs):
            return
        energy = _dot(y, y)
        raw = xs[0] if raw_predictor is None else raw_predictor.reshape(-1)
        raw_sse = max(0.0, energy + _dot(raw, raw) - 2.0 * _dot(y, raw))
        coefficients, fitted_sse = _least_squares(y, xs, energy)
        self.energy += energy
        self.raw_sse += raw_sse
        self.fitted_sse += fitted_sse
        self.count += 1
        self.elements += y.numel()
        self.coefficients.extend(coefficients)

    def summary(self) -> dict:
        return {
            "samples": self.count,
            "elements": self.elements,
            "raw_prediction_r2": _r2(self.raw_sse, self.energy),
            "fitted_prediction_r2": _r2(self.fitted_sse, self.energy),
            "coefficient_mean": _mean(self.coefficients),
            "coefficient_abs_mean": _mean([abs(x) for x in self.coefficients]),
        }


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left.reshape(-1), right.reshape(-1)))


def _least_squares(
    target: torch.Tensor, predictors: list[torch.Tensor], energy: float
) -> tuple[list[float], float]:
    if len(predictors) == 1:
        norm = _dot(predictors[0], predictors[0])
        coefficient = _dot(target, predictors[0]) / max(norm, 1e-30)
        sse = energy - coefficient * _dot(target, predictors[0])
        return [coefficient], max(0.0, sse)
    if len(predictors) != 2:
        raise ValueError("only one- and two-predictor fits are supported")
    first, second = predictors
    aa, bb, ab = _dot(first, first), _dot(second, second), _dot(first, second)
    ay, by = _dot(first, target), _dot(second, target)
    determinant = aa * bb - ab * ab
    if determinant <= 1e-20 * max(aa * bb, 1e-30):
        coefficient = ay / max(aa, 1e-30)
        return [coefficient, 0.0], max(0.0, energy - coefficient * ay)
    alpha = (ay * bb - by * ab) / determinant
    beta = (by * aa - ay * ab) / determinant
    sse = energy - alpha * ay - beta * by
    return [alpha, beta], max(0.0, sse)


def _r2(sse: float, energy: float) -> Optional[float]:
    return None if energy <= 0 else 1.0 - sse / energy


def _mean(values: list[float]) -> Optional[float]:
    return None if not values else sum(values) / len(values)


def _family(name: str) -> str:
    if "embedding" in name:
        return "embedding"
    if "self_attn" in name and name.endswith("weight"):
        return "attention_weight"
    if ("linear1" in name or "linear2" in name) and name.endswith("weight"):
        return "ffn_weight"
    if name.endswith("bias"):
        return "bias"
    if "norm" in name:
        return "normalization"
    return "other"


def _rank_one_energy(matrix: torch.Tensor, iterations: int) -> float:
    value = matrix.to(torch.float32)
    if value.ndim != 2 or not value.numel():
        return 0.0
    vector = torch.ones(value.shape[1], dtype=torch.float32)
    vector /= math.sqrt(vector.numel())
    for _ in range(iterations):
        vector = value.T.mv(value.mv(vector))
        norm = float(torch.linalg.vector_norm(vector))
        if norm == 0.0:
            return 0.0
        vector /= norm
    projected = value.mv(vector)
    return _dot(projected, projected)


def analyze_structure(
    trace_path: Path,
    *,
    sample_stride: int = 20,
    rank_stride: int = 200,
    rank_iterations: int = 4,
) -> dict:
    if sample_stride <= 0 or rank_stride <= 0 or rank_iterations <= 0:
        raise ValueError("analysis strides and iterations must be positive")
    lag1 = PredictionTotals()
    lag2 = PredictionTotals()
    adjacent = PredictionTotals()
    cross_layer = PredictionTotals()
    lag1_families: Dict[str, PredictionTotals] = {}
    previous: Dict[str, torch.Tensor] = {}
    previous_two: Dict[str, torch.Tensor] = {}
    rank_energy = 0.0
    rank_one_energy = 0.0
    rank_samples = 0
    sampled_steps = 0
    total_steps = 0

    for record in GradientTraceReader(Path(trace_path)).steps():
        total_steps += 1
        current = {
            name: tensor.detach().to(torch.float32).contiguous()
            for name, tensor in record.gradients.items()
            if tensor is not None
        }
        if record.step % sample_stride == 0:
            sampled_steps += 1
            for name, value in current.items():
                prior = previous.get(name)
                if prior is not None and prior.shape == value.shape:
                    lag1.add(value, [prior])
                    lag1_families.setdefault(_family(name), PredictionTotals()).add(
                        value, [prior]
                    )
                    older = previous_two.get(name)
                    if older is not None and older.shape == value.shape:
                        lag2.add(value, [prior, older])

                if value.ndim and value.shape[-1] > 1:
                    target = value[..., 1:]
                    predecessor = value[..., :-1]
                    adjacent.add(target, [predecessor])

            homologues: Dict[tuple[str, int], torch.Tensor] = {}
            for name, value in current.items():
                match = _LAYER.match(name)
                if match:
                    family = match.group(1) + "*" + match.group(3)
                    homologues[(family, int(match.group(2)))] = value
            for (family, layer), value in homologues.items():
                prior_layer = homologues.get((family, layer - 1))
                if prior_layer is not None and prior_layer.shape == value.shape:
                    cross_layer.add(value, [prior_layer])

        if record.step % rank_stride == 0:
            for value in current.values():
                if value.ndim != 2:
                    continue
                energy = _dot(value, value)
                rank_energy += energy
                rank_one_energy += _rank_one_energy(value, rank_iterations)
                rank_samples += 1

        previous_two, previous = previous, current

    return {
        "trace": str(Path(trace_path)),
        "configuration": {
            "sample_stride": sample_stride,
            "rank_stride": rank_stride,
            "rank_iterations": rank_iterations,
        },
        "steps": total_steps,
        "sampled_steps": sampled_steps,
        "temporal_lag1": lag1.summary(),
        "temporal_ar2": lag2.summary(),
        "temporal_lag1_by_family": {
            name: totals.summary() for name, totals in sorted(lag1_families.items())
        },
        "within_tensor_adjacent": adjacent.summary(),
        "within_step_cross_layer": cross_layer.summary(),
        "matrix_rank1": {
            "samples": rank_samples,
            "energy_fraction": (
                None if rank_energy <= 0 else rank_one_energy / rank_energy
            ),
            "power_iterations": rank_iterations,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=20)
    parser.add_argument("--rank-stride", type=int, default=200)
    parser.add_argument("--rank-iterations", type=int, default=4)
    args = parser.parse_args()
    experiment = Path(args.experiment)
    if not experiment.is_dir():
        root = Settings.load().experiments_dir
        experiment = root / (
            f"experiment_{int(args.experiment):03d}"
            if args.experiment.isdigit()
            else args.experiment
        )
    report = analyze_structure(
        experiment / "gradients.otsg",
        sample_stride=args.sample_stride,
        rank_stride=args.rank_stride,
        rank_iterations=args.rank_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
