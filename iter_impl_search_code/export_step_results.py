"""Export stable per-step compression tables from online metric JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "step",
    "source_tensor_bytes",
    "encoded_bytes",
    "compression_ratio",
    "compression_latency_ms",
    "compression_throughput_mib_per_second",
    "encoded_bits_per_gradient_element",
    "prediction_energy_r2",
    "prediction_cosine_similarity",
    "reconstruction_energy_r2",
)


def export_step_results(
    metrics_path: Path, output_path: Path, *, bytes_per_element: int = 4
) -> None:
    rows = []
    for line in Path(metrics_path).read_text(encoding="utf-8").splitlines():
        metric = json.loads(line)
        source_bytes = int(
            metric.get(
                "source_tensor_bytes",
                int(metric["gradient_elements"]) * bytes_per_element,
            )
        )
        encoded_bytes = int(metric["encoded_bytes"])
        seconds = float(metric["encode_seconds"])
        metric.update(
            {
                "source_tensor_bytes": source_bytes,
                "compression_ratio": source_bytes / encoded_bytes,
                "compression_latency_ms": 1_000.0 * seconds,
                "compression_throughput_mib_per_second": (
                    source_bytes / (1024.0 * 1024.0 * seconds)
                    if seconds
                    else None
                ),
            }
        )
        rows.append(metric)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bytes-per-element", type=int, default=4)
    args = parser.parse_args()
    export_step_results(
        args.metrics, args.output, bytes_per_element=args.bytes_per_element
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
