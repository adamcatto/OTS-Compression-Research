# D1 — OTS-DeltaQ v1

The exact runnable construction and result table are colocated in this bundle.

## Hypothesis

Per-block numeric approximation can reduce a FP32 gradient trace by about 4x
while preserving gradient energy-R2 above 0.99, and temporal prediction will
improve the residual representation without encoder/decoder drift.

## Method

The predictor for each tensor is its preceding *decoded* gradient. Residuals
are handled in 16,384-element blocks. A block uses nearest-rounded int8 with a
stored FP32 scale only when its squared reconstruction error is at most 1e-4 of
the original gradient block energy. It otherwise falls back to FP16 residuals,
then FP32. The per-block condition implies a whole-trace energy-R2 of at least
0.9999 up to floating-point arithmetic.

## Why this follows L0

L0 showed that generic lossless codecs save only ~7%. D1 is deliberately a
simple, decoder-consistent, independently sharded lossy baseline before adding
optimizer-aware rate allocation, learned prediction, or entropy coding.

## Gradient-screening result

On `experiment_001`, D1 produced a 9,648,672,495-byte artifact from the
19,491,431,115-byte source: 2.02012x compression. Encode/decode took 54.45 s
and 41.62 s. Gradient energy-R2 was 0.99999613 (relative squared error
3.87e-6) and cosine similarity was 0.99999807. This clears the 0.99 gate by a
large margin, but `experiment_001` predates saved initial/final weights, so it
cannot answer the optimizer replay question.

Artifact and metrics:
`<experiments>/experiment_001/lossy/ots_deltaq_v1/offline/`.

## Instrumented replay result

`experiment_002` repeated the exact same 1,000-step recipe while saving the
initial/final model states and replay contract. D1 again produced 2.02011x
compression (9,648,679,921 bytes), energy-R2 0.99999613, and cosine similarity
0.99999807. Applying its decoded gradients through a shadow AdamW replay gave:

| Measurement | Value |
| --- | ---: |
| Final-weight cosine similarity | 0.9999998842 |
| Final-weight relative L2 difference | 0.00048136 |
| Per-step loss MAE | 0.00001708 |
| Per-step loss RMSE | 0.00002460 |
| Maximum per-step loss difference | 0.00010419 |

The loss curve and replayed final state are stored under
`<experiments>/experiment_002/lossy/ots_deltaq_v1/offline/replay/`.

## What we learned

The strict per-block energy budget gives much more than the requested 0.99
gradient threshold and is sufficient for extremely close 1,000-step AdamW
replay on this model. Its cost is rate: the artifact is only 2.02x smaller,
not the nominal 4x int8 outcome, because many residual blocks correctly take
the FP16 fallback. The next iteration should compare an off-the-shelf lossy
control and then test optimizer-weighted allocation, rather than relaxing the
gate blindly.
