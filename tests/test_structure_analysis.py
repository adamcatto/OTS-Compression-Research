import torch

from ots_compression.implementation_search.structure_analysis import (
    PredictionTotals,
    _least_squares,
)


def test_scalar_fit_recovers_temporal_scale() -> None:
    previous = torch.arange(1, 9, dtype=torch.float32)
    current = 1.5 * previous
    totals = PredictionTotals()
    totals.add(current, [previous])

    summary = totals.summary()
    assert summary["fitted_prediction_r2"] == 1.0
    assert abs(summary["coefficient_mean"] - 1.5) < 1e-6


def test_ar2_fit_recovers_two_step_combination() -> None:
    first = torch.tensor([1.0, 0.0, 2.0, -1.0])
    second = torch.tensor([0.0, 1.0, -1.0, 2.0])
    target = 0.75 * first - 0.25 * second

    coefficients, sse = _least_squares(
        target, [first, second], float(torch.dot(target, target))
    )

    assert sse < 1e-6
    assert torch.allclose(torch.tensor(coefficients), torch.tensor([0.75, -0.25]))
