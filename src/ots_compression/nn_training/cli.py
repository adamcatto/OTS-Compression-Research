"""Command-line entry point for reproducible gradient-generation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from ..core.settings import Settings
from .config import load_experiment_spec
from .runner import run_training
from .tasks import default_task_registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--meta-config", type=Path)
    parser.add_argument("--experiment-number", type=int)
    parser.add_argument("--experiments-dir", type=Path)
    parser.add_argument("--datasets-dir", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    configured = Settings.load()
    settings = Settings(
        experiments_dir=arguments.experiments_dir or configured.experiments_dir,
        datasets_dir=arguments.datasets_dir or configured.datasets_dir,
    )
    spec = load_experiment_spec(
        arguments.config, meta_config_path=arguments.meta_config
    )
    task = default_task_registry().get(spec.task)
    result = run_training(
        spec,
        task,
        settings=settings,
        experiment_number=arguments.experiment_number,
    )
    print(
        json.dumps(
            {
                "completed_steps": result.completed_steps,
                "device": result.device,
                "experiment": str(result.experiment.path),
                "final_loss": result.final_loss,
                "parameter_count": result.parameter_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
