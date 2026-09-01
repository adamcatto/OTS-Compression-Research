import json
from pathlib import Path

import pytest
import torch

from ots_compression.compression.algorithms.ots_deltaq import OTSDeltaQCompressor
from ots_compression.compression.lossy import benchmark_deltaq
from ots_compression.core.gradient_trace import GradientTraceReader, GradientTraceWriter


def test_deltaq_round_trips_schema_and_meets_energy_budget(tmp_path: Path) -> None:
    source = tmp_path / "source.otsg"
    with GradientTraceWriter(source, experiment_id="unit") as writer:
        for step in range(3):
            writer.append(
                step,
                {
                    "weight": torch.randn(20_000) + step * 0.01,
                    "bias": torch.arange(7, dtype=torch.float32),
                    "unused": None,
                },
            )

    metrics = benchmark_deltaq(source, OTSDeltaQCompressor(), tmp_path / "lossy")

    assert metrics.compressed_bytes < metrics.source_bytes
    assert metrics.gradient_energy_r2 >= 0.999898
    assert metrics.gradient_cosine_similarity > 0.9999
    decoded_steps = list(GradientTraceReader(metrics.decoded_trace).steps())
    assert [step.step for step in decoded_steps] == [0, 1, 2]
    assert decoded_steps[0].gradients["unused"] is None


def test_deltaq_online_writer_saves_decoder_predictions_and_step_metrics(
    tmp_path: Path,
) -> None:
    compressor = OTSDeltaQCompressor(block_size=16, relative_squared_error=1e-4)
    artifact = tmp_path / "compressed.otsdq"
    predictions = tmp_path / "predictions.otsg"
    step_metrics = tmp_path / "prediction_metrics.jsonl"
    summary = tmp_path / "online_summary.json"
    gradients = [
        {"weight": torch.linspace(-1, 1, 32)},
        {"weight": torch.linspace(-0.9, 1.1, 32)},
    ]

    with compressor.open_online(
        artifact,
        source_metadata={
            "experiment_id": "online-unit",
            "format": "ots-gradient-trace",
            "gradient_capture": "optimizer_input",
        },
        predictions_path=predictions,
        prediction_metrics_path=step_metrics,
        summary_path=summary,
    ) as writer:
        for step, values in enumerate(gradients):
            writer.append(step, values)

    prediction_steps = list(GradientTraceReader(predictions).steps())
    decoded = tmp_path / "decoded.otsg"
    compressor.decompress(artifact, decoded)
    decoded_steps = list(GradientTraceReader(decoded).steps())
    metrics = [
        json.loads(line)
        for line in step_metrics.read_text(encoding="utf-8").splitlines()
    ]

    assert torch.count_nonzero(prediction_steps[0].gradients["weight"]) == 0
    assert torch.equal(
        prediction_steps[1].gradients["weight"],
        decoded_steps[0].gradients["weight"],
    )
    assert [metric["step"] for metric in metrics] == [0, 1]
    assert metrics[0]["prediction_energy_r2"] == 0.0
    assert metrics[1]["reconstruction_energy_r2"] >= 0.9999
    assert summary.is_file()


def test_deltaq_zero_predictor_is_decoder_consistent(tmp_path: Path) -> None:
    source = tmp_path / "source.otsg"
    with GradientTraceWriter(source, experiment_id="zero-unit") as writer:
        for step in range(3):
            writer.append(step, {"weight": torch.randn(20_000)})

    compressor = OTSDeltaQCompressor(prediction="zero")
    metrics = benchmark_deltaq(source, compressor, tmp_path / compressor.name)

    assert compressor.name == "ots_deltaq_zero_v1"
    assert compressor.configuration()["prediction"] == "zero"
    assert metrics.gradient_energy_r2 >= 0.999898
    assert metrics.codec_stats is not None
    assert metrics.codec_stats["steps"] == 3


def test_deltaq_rejects_unknown_predictor() -> None:
    with pytest.raises(ValueError, match="prediction"):
        OTSDeltaQCompressor(prediction="future_gradient")
