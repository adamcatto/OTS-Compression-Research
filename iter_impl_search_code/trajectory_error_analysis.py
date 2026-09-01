"""Measure signed error accumulation in a decoded gradient trajectory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ots_compression.core.gradient_trace import GradientTraceReader


def analyze_trajectory_error(source: Path, decoded: Path) -> dict:
    source_steps = GradientTraceReader(Path(source)).steps()
    decoded_steps = GradientTraceReader(Path(decoded)).steps()
    cumulative_error: dict[str, torch.Tensor] = {}
    cumulative_gradient: dict[str, torch.Tensor] = {}
    steps = values = 0
    error_sum = error_energy = gradient_energy = 0.0
    for exact_step, lossy_step in zip(source_steps, decoded_steps):
        if (
            exact_step.step != lossy_step.step
            or exact_step.gradients.keys() != lossy_step.gradients.keys()
        ):
            raise ValueError("decoded trace schema differs from source")
        for name, exact in exact_step.gradients.items():
            lossy = lossy_step.gradients[name]
            if exact is None and lossy is None:
                continue
            if exact is None or lossy is None or exact.shape != lossy.shape:
                raise ValueError(f"decoded gradient mismatch for {name}")
            exact64 = exact.to(torch.float64)
            error64 = lossy.to(torch.float64) - exact64
            if name not in cumulative_error:
                cumulative_error[name] = torch.zeros_like(error64)
                cumulative_gradient[name] = torch.zeros_like(exact64)
            cumulative_error[name].add_(error64)
            cumulative_gradient[name].add_(exact64)
            values += exact.numel()
            error_sum += float(error64.sum())
            error_energy += float(error64.square().sum())
            gradient_energy += float(exact64.square().sum())
        steps += 1
    if next(source_steps, None) is not None or next(decoded_steps, None) is not None:
        raise ValueError("decoded trace has a different step count")

    cumulative_error_energy = sum(
        float(value.square().sum()) for value in cumulative_error.values()
    )
    cumulative_gradient_energy = sum(
        float(value.square().sum()) for value in cumulative_gradient.values()
    )
    tensors = []
    for name, error in cumulative_error.items():
        error_l2 = float(torch.linalg.vector_norm(error))
        gradient_l2 = float(torch.linalg.vector_norm(cumulative_gradient[name]))
        tensors.append(
            {
                "name": name,
                "cumulative_error_l2": error_l2,
                "cumulative_gradient_l2": gradient_l2,
                "cumulative_error_relative_l2": error_l2
                / max(gradient_l2, 1e-30),
            }
        )
    tensors.sort(key=lambda row: row["cumulative_error_l2"], reverse=True)
    return {
        "steps": steps,
        "values": values,
        "gradient_energy_r2": 1.0 - error_energy / max(gradient_energy, 1e-30),
        "scalar_error_mean": error_sum / max(values, 1),
        "scalar_error_rms": math.sqrt(error_energy / max(values, 1)),
        "cumulative_error_l2": math.sqrt(cumulative_error_energy),
        "cumulative_gradient_l2": math.sqrt(cumulative_gradient_energy),
        "cumulative_error_relative_l2": math.sqrt(
            cumulative_error_energy / max(cumulative_gradient_energy, 1e-30)
        ),
        "cumulative_error_rms_per_parameter": math.sqrt(
            cumulative_error_energy
            / max(sum(value.numel() for value in cumulative_error.values()), 1)
        ),
        "largest_cumulative_error_tensors": tensors[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("decoded", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_trajectory_error(args.source, args.decoded)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
