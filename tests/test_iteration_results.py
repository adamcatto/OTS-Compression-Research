import csv
import json
from pathlib import Path

from iter_impl_search_code.export_final_model_results import (
    export_final_model_results,
)
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


def test_export_final_model_results_uses_stable_schema(tmp_path: Path) -> None:
    source = tmp_path / "metrics.json"
    source.write_text(
        json.dumps(
            {
                "evaluation_split": "validation",
                "evaluated_batches": 2,
                "evaluated_predictions": 8,
                "reference_label_cross_entropy": 1.0,
                "lossy_label_cross_entropy": 1.01,
                "label_cross_entropy_delta": 0.01,
                "reference_to_lossy_cross_entropy": 1.2,
                "reference_to_lossy_kl": 0.002,
                "lossy_to_reference_kl": 0.003,
                "jensen_shannon_divergence": 0.0005,
                "top1_agreement": 0.875,
                "mean_absolute_logit_difference": 0.04,
                "logit_relative_l2": 0.02,
                "mean_absolute_probability_difference": 0.001,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "final_model_outputs.csv"

    export_final_model_results("E2", source, output)

    with output.open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["algorithm_id"] == "E2"
    assert row["reference_to_lossy_cross_entropy"] == "1.2"
    assert row["reference_to_lossy_kl"] == "0.002"
    assert row["top1_agreement"] == "0.875"
