from pathlib import Path

import pytest
import torch

from ots_compression.core.gradient_trace import GradientTraceReader, GradientTraceWriter


def test_gradient_trace_round_trip_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "gradients.otsg"
    expected_weight = torch.tensor([[1.0, -2.0], [3.5, 4.0]], dtype=torch.float32)
    expected_bias = torch.tensor([1.25, -0.5], dtype=torch.bfloat16)

    with GradientTraceWriter(path, experiment_id="experiment_001") as writer:
        writer.append(0, {"weight": expected_weight, "unused": None})
    with GradientTraceWriter(path, experiment_id="experiment_001") as writer:
        writer.append(1, {"weight": expected_weight + 1, "bias": expected_bias})

    steps = list(GradientTraceReader(path).steps())
    assert [step.step for step in steps] == [0, 1]
    assert torch.equal(steps[0].gradients["weight"], expected_weight)
    assert steps[0].gradients["unused"] is None
    assert torch.equal(steps[1].gradients["bias"], expected_bias)
    assert b"".join(
        [GradientTraceReader(path).raw_header()]
        + list(GradientTraceReader(path).iter_raw_records())
    ) == path.read_bytes()


def test_steps_must_increase(tmp_path: Path) -> None:
    path = tmp_path / "gradients.otsg"
    with GradientTraceWriter(path, experiment_id="experiment_001") as writer:
        writer.append(3, {"gradient": torch.ones(2)})
        with pytest.raises(ValueError, match="strictly increasing"):
            writer.append(3, {"gradient": torch.ones(2)})
