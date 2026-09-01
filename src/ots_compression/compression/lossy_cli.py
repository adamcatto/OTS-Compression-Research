"""Run OTS-DeltaQ against a completed gradient trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.settings import Settings
from .algorithms.ots_deltaq import OTSDeltaQCompressor
from .lossy import benchmark_deltaq, evaluate_deltaq_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--block-size", type=int, default=16_384)
    parser.add_argument("--relative-squared-error", type=float, default=1e-4)
    parser.add_argument("--outlier-fraction", type=float, default=0.0)
    parser.add_argument("--rank1-power-iterations", type=int, default=1)
    parser.add_argument("--rank1-warm-start", action="store_true")
    parser.add_argument(
        "--allocation",
        choices=["per_block", "optimizer_aware_global"],
        default="per_block",
    )
    parser.add_argument(
        "--preconditioned-relative-squared-error", type=float, default=1e-4
    )
    parser.add_argument("--sensitivity-beta2", type=float, default=0.999)
    parser.add_argument("--sensitivity-epsilon", type=float, default=1e-8)
    parser.add_argument("--allocator-iterations", type=int, default=40)
    parser.add_argument(
        "--block-transform",
        choices=["none", "randomized_hadamard"],
        default="none",
    )
    parser.add_argument(
        "--prediction",
        choices=[
            "previous_decoded_gradient",
            "zero",
            "adaptive_zero_previous",
            "rank1_tensor",
        ],
        default="previous_decoded_gradient",
    )
    parser.add_argument(
        "--access-label",
        choices=["offline", "causal-posthoc"],
        default="offline",
        help="Result label for a completed-trace run; does not change causality.",
    )
    parser.add_argument(
        "--decode-online",
        action="store_true",
        help="Decode and score the artifact emitted inside the training loop.",
    )
    args = parser.parse_args()
    root = Settings.load().experiments_dir
    experiment = Path(args.experiment)
    if not experiment.is_dir():
        experiment = root / (f"experiment_{int(args.experiment):03d}" if args.experiment.isdigit() else args.experiment)
    compressor = OTSDeltaQCompressor(
        block_size=args.block_size,
        relative_squared_error=args.relative_squared_error,
        prediction=args.prediction,
        outlier_fraction=args.outlier_fraction,
        rank1_power_iterations=args.rank1_power_iterations,
        rank1_warm_start=args.rank1_warm_start,
        block_transform=args.block_transform,
        allocation=args.allocation,
        preconditioned_relative_squared_error=(
            args.preconditioned_relative_squared_error
        ),
        sensitivity_beta2=args.sensitivity_beta2,
        sensitivity_epsilon=args.sensitivity_epsilon,
        allocator_iterations=args.allocator_iterations,
    )
    mode = (
        "online"
        if args.decode_online
        else args.access_label.replace("-", "_")
    )
    destination = experiment / "lossy" / compressor.name / mode
    if args.decode_online:
        step_metrics = destination / "prediction_metrics.jsonl"
        encode_seconds = sum(
            json.loads(line)["encode_seconds"]
            for line in step_metrics.read_text(encoding="utf-8").splitlines()
        )
        metrics = evaluate_deltaq_artifact(
            experiment / "gradients.otsg",
            compressor,
            destination / "compressed.otsdq",
            destination,
            encode_seconds=encode_seconds,
        )
    else:
        metrics = benchmark_deltaq(
            experiment / "gradients.otsg", compressor, destination
        )
    print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
