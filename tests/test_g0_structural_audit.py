import math

import pytest
import torch

from iter_impl_search_code.experiments.G0.algorithm import (
    _tensor_train_approximation,
    gaussian_residual_rate_lower_bound,
    kronecker_rearrangement,
)


def test_kronecker_rearrangement_makes_exact_product_rank_one() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.arange(1.0, 17.0).reshape(4, 4)
    matrix = torch.kron(left, right)

    singular_values = torch.linalg.svdvals(kronecker_rearrangement(matrix))
    energy_fraction = singular_values[0].square() / singular_values.square().sum()

    assert energy_fraction == pytest.approx(1.0, abs=1e-6)


def test_gaussian_rate_bound_matches_r2_budget() -> None:
    assert gaussian_residual_rate_lower_bound(0.01) == 0.0
    assert gaussian_residual_rate_lower_bound(1.0) == pytest.approx(
        0.5 * math.log2(100.0)
    )
    threshold = 0.01 * 2.0**0.64
    assert gaussian_residual_rate_lower_bound(threshold) == pytest.approx(0.32)


def test_tensor_train_svd_reconstructs_low_rank_tensorization() -> None:
    vectors = [torch.tensor([1.0, 2.0]), torch.tensor([2.0, -1.0])] * 2
    matrix = torch.einsum("i,j,k,l->ijkl", *vectors).reshape(4, 4)

    reconstructed, parameters = _tensor_train_approximation(matrix, (2, 2, 2, 2), 1)

    assert torch.allclose(reconstructed, matrix, atol=1e-4)
    assert parameters == 8
