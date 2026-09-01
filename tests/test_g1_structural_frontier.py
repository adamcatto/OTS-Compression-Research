import torch

from iter_impl_search_code.experiments.G1.algorithm import (
    _decode_array,
    _encode_array,
    _pack_codes,
    _unpack_codes,
    run_step,
)


def test_bit_packing_round_trips_supported_widths() -> None:
    values = torch.arange(37).numpy()
    for bits in (1, 2, 4, 8, 16):
        maximum = (1 << bits) - 1
        expected = values % (maximum + 1)
        packed = _pack_codes(expected, bits)
        decoded = _unpack_codes(packed, bits, len(expected))
        assert torch.equal(torch.from_numpy(decoded), torch.from_numpy(expected).float())


def test_quantized_array_round_trip_matches_encoder_reconstruction() -> None:
    value = torch.linspace(-2.0, 3.0, 117).reshape(9, 13)
    for bits in (1, 2, 4, 8, 16):
        payload, metadata, expected = _encode_array(value, bits)
        actual = _decode_array(payload, metadata)
        assert torch.equal(actual, expected)


def test_actual_stream_respects_budget_and_decodes() -> None:
    torch.manual_seed(7)
    left = torch.randn(16, 4)
    right = torch.randn(4, 16)
    previous = {"weight": left @ right, "bias": torch.randn(16)}
    current = {
        "weight": previous["weight"] + 0.01 * torch.randn(16, 16),
        "bias": previous["bias"] + 0.01 * torch.randn(16),
    }
    row, stream = run_step(current, previous, budget_bits_per_value=64.0)

    assert len(stream) == row["compressed_bytes"]
    assert len(stream) <= row["budget_bytes"]
    assert 0.0 <= row["gradient_energy_r2"] <= 1.0
    assert row["tensor_count"] == 2
