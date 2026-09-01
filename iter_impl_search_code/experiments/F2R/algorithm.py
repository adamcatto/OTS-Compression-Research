"""Exact compressor construction for F2R on experiment_003."""

from ots_compression.compression.algorithms.ots_deltaq import OTSDeltaQCompressor


# This is F1's measured payload rate, excluding record/file metadata:
# 8 * 5,644,610,904 payload bytes / 4,870,400,000 gradient values.
F1_PAYLOAD_BITS_PER_ELEMENT = 9.27169990801577


def build_compressor() -> OTSDeltaQCompressor:
    return OTSDeltaQCompressor(
        block_size=16_384,
        relative_squared_error=1e-4,
        prediction="zero",
        block_transform="randomized_hadamard",
        allocation="rate_matched_optimizer",
        sensitivity_beta2=0.95,
        sensitivity_epsilon=1e-8,
        target_payload_bits_per_element=F1_PAYLOAD_BITS_PER_ELEMENT,
    )
