"""Compare reference and lossy-replay final-model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.settings import Settings
from .final_model_eval import evaluate_saved_final_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--replayed-model", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=32)
    args = parser.parse_args()
    experiment = Path(args.experiment)
    if not experiment.is_dir():
        root = Settings.load().experiments_dir
        experiment = root / (
            f"experiment_{int(args.experiment):03d}"
            if args.experiment.isdigit()
            else args.experiment
        )
    metrics = evaluate_saved_final_models(
        experiment,
        args.replayed_model,
        args.destination,
        max_batches=args.batches,
    )
    print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
