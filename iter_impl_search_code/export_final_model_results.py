"""Export one stable final-model output-comparison row from replay metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "algorithm_id",
    "evaluation_split",
    "evaluated_batches",
    "evaluated_predictions",
    "reference_label_cross_entropy",
    "lossy_label_cross_entropy",
    "label_cross_entropy_delta",
    "reference_to_lossy_cross_entropy",
    "reference_to_lossy_kl",
    "lossy_to_reference_kl",
    "jensen_shannon_divergence",
    "top1_agreement",
    "mean_absolute_logit_difference",
    "logit_relative_l2",
    "mean_absolute_probability_difference",
)


def export_final_model_results(
    algorithm_id: str, metrics_path: Path, output_path: Path
) -> None:
    row = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    row["algorithm_id"] = algorithm_id
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm_id")
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_final_model_results(args.algorithm_id, args.metrics, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
