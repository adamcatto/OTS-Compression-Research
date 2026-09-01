import json
from pathlib import Path

import pytest
import torch

from ots_compression.compression.algorithms.ots_deltaq import OTSDeltaQCompressor
from ots_compression.compression.lossy import (
    benchmark_deltaq,
    compare_prediction_traces,
)
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
    assert metrics[1]["compression_latency_ms"] > 0
    assert metrics[1]["compression_ratio"] > 0
    assert metrics[1]["compression_throughput_mib_per_second"] > 0
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


def test_deltaq_adaptive_predictor_selects_per_block(tmp_path: Path) -> None:
    block_size = 1_024
    compressor = OTSDeltaQCompressor(
        block_size=block_size,
        prediction="adaptive_zero_previous",
    )
    current = torch.full((2 * block_size,), 0.0039)
    previous = torch.zeros_like(current)
    current[0] = 1.0
    previous[0] = 1.0
    previous[block_size] = 1.0

    _, reconstruction, selected_prediction, counts = compressor._encode_tensor(
        current, previous
    )

    assert compressor.name == "ots_deltaq_adaptive_predictor_v1"
    assert counts["previous_prediction_blocks"] == 1
    assert counts["zero_prediction_blocks"] == 1
    assert selected_prediction[0] == 1.0
    assert selected_prediction[block_size] == 0.0
    relative_error = float((current - reconstruction).square().sum() / current.square().sum())
    assert relative_error <= 1e-4

    source = tmp_path / "adaptive.otsg"
    with GradientTraceWriter(source, experiment_id="adaptive-unit") as writer:
        writer.append(0, {"weight": previous})
        writer.append(1, {"weight": current})
    metrics = benchmark_deltaq(
        source, compressor, tmp_path / "ots_deltaq_adaptive_predictor_v1"
    )
    assert metrics.gradient_energy_r2 >= 0.999898
    assert metrics.codec_stats["previous_prediction_blocks"] > 0
    assert metrics.codec_stats["zero_prediction_blocks"] > 0


def test_deltaq_sparse_outliers_enable_int8_inliers(tmp_path: Path) -> None:
    block_size = 1_024
    compressor = OTSDeltaQCompressor(
        block_size=block_size,
        prediction="zero",
        outlier_fraction=1 / block_size,
    )
    gradient = torch.full((block_size,), 0.0039)
    gradient[0] = 1.0

    _, reconstruction, _, counts = compressor._encode_tensor(
        gradient, torch.zeros_like(gradient)
    )

    assert compressor.name == "ots_deltaq_outlier_v1"
    assert counts["int8_outlier_blocks"] == 1
    assert counts["outlier_values"] == 1
    relative_error = float(
        (gradient - reconstruction).square().sum() / gradient.square().sum()
    )
    assert relative_error <= 1e-4

    source = tmp_path / "outlier.otsg"
    with GradientTraceWriter(source, experiment_id="outlier-unit") as writer:
        writer.append(0, {"weight": gradient})
    metrics = benchmark_deltaq(source, compressor, tmp_path / compressor.name)
    assert metrics.gradient_energy_r2 >= 0.9999
    assert metrics.codec_stats["int8_outlier_elements"] == block_size


def test_deltaq_rejects_invalid_outlier_fraction() -> None:
    with pytest.raises(ValueError, match="outlier_fraction"):
        OTSDeltaQCompressor(outlier_fraction=1.0)


def test_rank1_tensor_prediction_round_trips_selected_factors(tmp_path: Path) -> None:
    left = torch.logspace(-3, 0, 128)
    right = torch.logspace(-3, 0, 128)
    gradient = torch.outer(left, right)
    source = tmp_path / "rank1.otsg"
    with GradientTraceWriter(source, experiment_id="rank1-unit") as writer:
        writer.append(0, {"matrix": gradient})

    compressor = OTSDeltaQCompressor(
        block_size=16_384,
        prediction="rank1_tensor",
        rank1_power_iterations=1,
    )
    metrics = benchmark_deltaq(source, compressor, tmp_path / compressor.name)

    assert metrics.gradient_energy_r2 >= 0.9999
    assert metrics.codec_stats["rank1_prediction_tensors"] == 1
    assert metrics.codec_stats["rank1_factor_bytes"] == 512
    predictions = list(
        GradientTraceReader(
            tmp_path / compressor.name / "predictions.otsg"
        ).steps()
    )
    assert len(predictions) == 1
    assert predictions[0].gradients["matrix"].shape == gradient.shape
    extracted_path = tmp_path / "extracted_predictions.otsg"
    compressor.extract_predictions(
        tmp_path / compressor.name / "compressed.otsdq", extracted_path
    )
    extracted = list(GradientTraceReader(extracted_path).steps())
    assert torch.equal(
        predictions[0].gradients["matrix"], extracted[0].gradients["matrix"]
    )
    comparison_path = tmp_path / "prediction_comparison.jsonl"
    compare_prediction_traces(source, extracted_path, comparison_path)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["step"] == 0
    assert comparison["prediction_energy_r2"] > 0.999


def test_rank1_rejects_invalid_iteration_count() -> None:
    with pytest.raises(ValueError, match="rank1_power_iterations"):
        OTSDeltaQCompressor(rank1_power_iterations=0)


def test_e2_warm_start_tracks_temporal_matrix_subspace() -> None:
    compressor = OTSDeltaQCompressor(
        prediction="rank1_tensor",
        rank1_power_iterations=1,
        rank1_warm_start=True,
    )
    first = torch.diag(torch.tensor([10.0, 1.0]))
    current = torch.diag(torch.tensor([10.0, 9.0]))

    initial = compressor._encode_rank1_prediction(first)
    assert initial is not None
    warm = compressor._encode_rank1_prediction(current, initial[2])
    fixed = compressor._encode_rank1_prediction(current)
    assert warm is not None and fixed is not None
    warm_error = float((current - warm[0]).square().sum())
    fixed_error = float((current - fixed[0]).square().sum())

    assert compressor.name == "ots_rank1_tracking_e2"
    assert compressor.configuration()["rank1_warm_start"] is True
    assert warm_error < fixed_error
