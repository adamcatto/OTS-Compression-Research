# D2 — training-time online DeltaQ with saved predictions

## Hypothesis

D1's causal result should reproduce when compression is performed synchronously
inside training, immediately after each optimizer-input gradient snapshot is
stored. Saving the actual decoder-side prediction tensors and scalar metrics
will reveal whether prediction quality changes over training and whether the
previous-decoded-gradient predictor is doing useful work.

## Why this follows D1

D1 processed a completed trace even though it obeyed causal predictor rules.
That established rate and replay fidelity but did not measure training-time
overhead or prove the append/checkpoint behavior. D2 closes that methodological
gap without changing the quantizer, so differences can be attributed to online
integration rather than a new compression method.

## Method

At every configured gradient-capture point, the runner synchronously:

1. appends the exact gradient snapshot to `gradients.otsg`;
2. predicts from the previous decoded gradient only;
3. appends one complete DeltaQ record to `compressed.otsdq` and flushes it;
4. appends the prediction tensors to `predictions.otsg`; and
5. records prediction energy-R2/cosine, reconstruction energy-R2, encoded bytes,
   block modes, and codec time in `prediction_metrics.jsonl`.

Predictions are evaluation instrumentation and are not counted as compressed
side information because the decoder recreates them from prior decoded values.
They are stored only so prediction behavior can be plotted and inspected.

## Results

`experiment_003` completed the same 1,000-step WikiText-2 run while emitting
DeltaQ records synchronously. Training finished in 438.43 s with final loss
2.1219473. The exact trace remained 19,491,431,115 bytes; the online artifact
was 9,648,640,266 bytes (2.02012x compression).

| Compression/replay measurement | Value |
| --- | ---: |
| Summed codec time | 48.94 s (48.94 ms/step) |
| Decoder time | 42.08 s |
| Gradient energy-R2 | 0.9999961290 |
| Gradient cosine similarity | 0.9999980645 |
| Final-weight cosine similarity | 0.9999998838 |
| Final-weight relative L2 difference | 0.00048205 |
| Replay loss MAE | 0.00001698 |
| Replay loss RMSE | 0.00002456 |
| Maximum replay loss difference | 0.00010538 |

All 1,000 training steps have a corresponding compressed record, saved
prediction tensor record, and prediction-metrics row. The actual predictions
occupy a diagnostic `predictions.otsg` trace and are not counted in compressed
size or needed for decoding.

## Prediction behavior

| Prediction measurement across steps | Value |
| --- | ---: |
| Mean / median prediction energy-R2 | -0.4719 / -0.5121 |
| Prediction energy-R2 5th / 95th percentile | -1.2868 / 0.8002 |
| Steps with prediction energy-R2 > 0 | 132 / 1,000 |
| Mean / median prediction cosine | 0.2765 / 0.2449 |
| Mean encoded bits per gradient element | 15.8486 |
| Int8 / FP16 / FP32 blocks | 39,137 / 306,814 / 1,049 |

The full trajectory is in `prediction_metrics.jsonl`, and aggregate plot-ready
statistics are in `online_summary.json`, under
`<experiments>/experiment_003/lossy/ots_deltaq_v1/online/`.

## What we learned

True online execution reproduces D1's compression and downstream fidelity, so
post-hoc causality was not hiding a correctness issue. The surprising result
is that the previous decoded minibatch gradient is usually worse than predicting
zero. Its residual has larger scale relative to the current-gradient energy,
which forces 88.4% of blocks into FP16 and limits compression to ~2x.

The next iteration should first implement the zero-predictor ablation, which may
allow more direct-gradient blocks to pass int8. Then compare a decoder-available
low-pass or optimizer-moment predictor. This evidence takes priority over adding
more sophisticated entropy coding to the current poor residual.
