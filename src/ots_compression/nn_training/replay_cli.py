"""Replay a decoded gradient trace through an experiment's saved optimizer setup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.settings import Settings
from .replay import replay_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--gradient-trace", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = Settings.load().experiments_dir
    experiment = Path(args.experiment)
    if not experiment.is_dir():
        experiment = root / (f"experiment_{int(args.experiment):03d}" if args.experiment.isdigit() else args.experiment)
    print(json.dumps(replay_experiment(experiment, Path(args.gradient_trace), args.destination).to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
