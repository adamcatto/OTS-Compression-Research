# D5 — sparse outlier-aware residual quantization

The exact runnable construction and result table are colocated in this bundle.

## Hypothesis

The max residual in a block is forcing otherwise quantizable values into FP16.
Preserving the largest 1% of residual magnitudes as sparse FP16 exceptions and
scaling dense int8 from the remaining inliers should materially improve rate
while retaining D2's previous-decoded predictor and the same error gate.

## Why this follows D4

D4 showed that predictor selection moves rate by only 0.19%, while 96.4% of
elements remain FP16. The quantizer's sensitivity to block maxima is now the
dominant measured failure mode, so D5 changes quantization rather than adding
more predictor state.

## Method and access regime

For each residual block, D5 evaluates the existing int8/FP16/FP32 fallback and
one robust candidate. The candidate removes the largest 1% magnitudes, int8
quantizes the dense inliers, and stores each exception as an int32 index plus
FP16 value. It is selected only when it meets the unchanged 1e-4 block error
budget and has fewer payload bytes. Format version 3 adds the sparse mode while
retaining version-1/2 decode compatibility.

The algorithm is online-capable, linear apart from local top-k selection,
shard-local, and bounded to one block. Its first controlled measurement reuses
`experiment_003` and is labeled `causal_posthoc`.

## Results

The causal-posthoc run completed under
`<experiments>/experiment_003/lossy/ots_deltaq_outlier_v1/causal_posthoc/`.

| Metric | D5 |
| --- | ---: |
| Source bytes | 19,491,431,115 |
| Compressed bytes | 6,298,265,295 |
| Compression ratio | 3.09473x |
| Gradient energy R2 | 0.999952017 |
| Gradient cosine | 0.999976009 |
| Encode time | 104.11 s |
| Decode time | 42.52 s |
| Replay loss MAE | 5.649e-5 |
| Replay loss RMSE | 1.234e-4 |
| Final-weight cosine | 0.999595629 |
| Final-weight relative L2 | 0.0286023 |

Of 4.87 billion values, 3.552 billion used the sparse-outlier mode and
35.56 million values were stored as exceptions. Another 1.170 billion values
fell back to FP16. This increased compression by about 53% over D2's 2.020x,
while remaining comfortably above the requested 0.99 gradient-R2 floor.

## Final-model output agreement

On 131,072 held-out next-token predictions, the replayed model has 99.7200%
top-1 agreement with the normal final model. Reference-to-lossy KL is
5.062e-5, label cross-entropy changes by +1.434e-4, and logit relative L2 is
11.7954%. The complete comparison is in `final_model_outputs.csv`.

## Interpretation

The loss trajectory is close enough for many training-telemetry or approximate
replay applications: its mean absolute loss deviation is only 5.65e-5. The
final parameter vector tells a stricter story. A 2.86% relative-L2 deviation is
far larger than D2's 0.0482%, even though the direction remains close. Sparse
outlier quantization is therefore a strong rate/trajectory tradeoff, but not a
replacement for D2 when nearly identical final weights are required.

The result also shows why aggregate gradient R2 cannot be the only acceptance
test. Small, biased errors can accumulate through 1,000 optimizer steps without
visibly disrupting the scalar loss curve. Future candidates must report both
loss-curve and final-weight agreement, and should test error-feedback or
explicitly bias-controlled residual coding when pushing toward D5's rate.

## Next decision

Do not continue tuning the outlier fraction immediately. Start a distinct E
lineage that measures and exploits tensor structure. E1 should first quantify
temporal linear predictability and within-matrix structure, then implement the
smallest decoder-reproducible learned predictor supported by those measurements.
