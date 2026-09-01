import csv
import json
from pathlib import Path

from iter_impl_search_code.export_step_results import export_step_results


def test_export_step_results_derives_online_rate_columns(tmp_path: Path) -> None:
    source = tmp_path / "metrics.jsonl"
    source.write_text(
        json.dumps(
            {
                "step": 7,
                "gradient_elements": 100,
                "encoded_bytes": 200,
                "encode_seconds": 0.01,
                "encoded_bits_per_gradient_element": 16.0,
                "prediction_energy_r2": 0.5,
                "prediction_cosine_similarity": 0.9,
                "reconstruction_energy_r2": 0.999,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "results.csv"

    export_step_results(source, output)

    with output.open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["step"] == "7"
    assert row["source_tensor_bytes"] == "400"
    assert row["compression_ratio"] == "2.0"
    assert row["compression_latency_ms"] == "10.0"
