import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from iter_impl_search_code.experiments.G1P.algorithm import _batch, _compare, sampler_schedule


def test_stored_schedule_matches_single_worker_dataloader_prefix() -> None:
    dataset = TensorDataset(torch.arange(400), torch.arange(400) + 1)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(17),
        drop_last=True,
    )
    schedule = sampler_schedule(len(dataset), 8, 5, 17)
    iterator = iter(loader)
    for indices in schedule:
        expected = next(iterator)
        actual = _batch(dataset, indices)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


def test_gradient_comparison_detects_exact_and_inexact_replay() -> None:
    exact = {"a": torch.tensor([1.0, -2.0]), "b": torch.tensor([3.0])}
    identical = _compare(exact, exact)
    assert identical["bit_exact"] is True
    assert identical["gradient_energy_r2"] == 1.0
    assert identical["relative_l2_error"] == 0.0

    changed = {"a": exact["a"] + torch.tensor([0.1, 0.0]), "b": exact["b"]}
    metrics = _compare(changed, exact)
    assert metrics["bit_exact"] is False
    assert 0.99 < metrics["gradient_energy_r2"] < 1.0


def test_schedule_is_explicit_little_endian_uint32() -> None:
    schedule = sampler_schedule(100, 4, 3, 2)
    assert schedule.shape == (3, 4)
    assert schedule.dtype == np.dtype("<u4")
    assert len(np.unique(schedule)) == schedule.size
