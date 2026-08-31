"""Run established compression baselines against a completed experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from ..core.settings import Settings
from .algorithms.baselines import baseline_registry
from .api import AccessMode
from .evaluation import benchmark_trace, write_metrics, write_metrics_table


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment directory, experiment_001, or numeric experiment ID",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        help="Baseline name; repeat as needed. Defaults to every installed codec.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=[mode.value for mode in AccessMode],
        help="Access regime; repeat as needed. Defaults to online and offline.",
    )
    return parser


def _experiment_path(value: str, root: Path) -> Path:
    supplied = Path(value)
    if supplied.is_dir():
        return supplied
    if value.isdigit():
        supplied = root / f"experiment_{int(value):03d}"
    elif value.startswith("experiment_"):
        supplied = root / value
    if not supplied.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {supplied}")
    return supplied


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    experiment = _experiment_path(
        arguments.experiment, Settings.load().experiments_dir
    )
    trace = experiment / "gradients.otsg"
    if not trace.is_file():
        raise FileNotFoundError(f"gradient trace does not exist: {trace}")

    registry = baseline_registry()
    names = arguments.algorithm or sorted(registry)
    missing = sorted(set(names) - set(registry))
    if missing:
        raise RuntimeError(
            "baseline unavailable or unknown: "
            + ", ".join(missing)
            + "; installed baselines: "
            + ", ".join(sorted(registry))
        )
    modes = [AccessMode(value) for value in (arguments.mode or ["online", "offline"])]

    results = []
    metrics_results = []
    for name in names:
        compressor = registry[name]
        for mode in modes:
            destination = experiment / "baselines" / name / mode.value
            artifact = destination / "compressed.otsc"
            metrics = benchmark_trace(
                trace, compressor, mode, artifact_path=artifact
            )
            metrics_path = destination / "metrics.json"
            write_metrics(metrics, metrics_path)
            metrics_results.append(metrics)
            results.append(
                {
                    "algorithm": name,
                    "artifact": str(artifact),
                    "compression_ratio": metrics.compression_ratio,
                    "metrics": str(metrics_path),
                    "mode": mode.value,
                }
            )
    write_metrics_table(metrics_results, experiment / "baselines")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
