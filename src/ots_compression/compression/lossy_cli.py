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
    parser.add_argument(
        "--prediction",
        choices=["previous_decoded_gradient", "zero"],
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
