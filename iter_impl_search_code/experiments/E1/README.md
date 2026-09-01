# E1 — learned rank-1 tensor structure

The exact runnable construction and result table are colocated in this bundle.

## Hypothesis

Large weight-gradient matrices contain a coherent low-rank component that is
more useful than copying a noisy minibatch gradient through time. Sending a
small learned rank-1 component and strictly quantizing its residual should move
more values from FP16 to int8 without adopting D5's looser sparse-outlier path.

## Why this starts a new lineage

D2-D4 established that direct temporal predictors have poor aggregate quality,
and D5 showed that relaxing the residual representation can reach 3.095x but
accumulate a 2.86% final-weight difference. E1 returns to D2's strict 1e-4
per-block relative-SSE gate and explores model structure rather than another
outlier-policy increment. The `E` prefix records that departure.

## Structure study

`implementation_search/structure_analysis.py` streamed the existing exact
trace once, sampling every 20th step for prediction statistics and every 200th
step for rank analysis. The report is stored outside git at
`<experiments>/experiment_003/lossy/ots_structural_e1/analysis/structure_report.json`.

| Candidate structure | Energy-weighted fitted prediction R2 |
| --- | ---: |
| Adjacent coordinates within a tensor | 0.00195 |
| Homologous tensor from the preceding layer | 0.05908 |
| Per-tensor fitted temporal lag 1 | 0.17596 |
| Per-tensor fitted temporal AR(2) | 0.21841 |
| Rank-1 component of weight matrices | 0.59466 energy fraction |

Raw adjacent and cross-layer copies were actively harmful. AR(2) was useful but
substantially weaker than the matrix component. One power iteration captured
96.5% of matrix energy at step zero, nearly matching four iterations there;
the full sampled trace shows that this structure weakens later but remains the
strongest candidate.

## Method and scaling properties

For every two-dimensional gradient shard, the encoder starts from a fixed
all-ones right vector and performs one power iteration. It converts the learned
left and right factors to FP16, reconstructs those decoded factors locally, and
feeds the resulting rank-1 prediction to the existing int8/FP16/FP32 DeltaQ
residual codec. It also encodes the ordinary previous-decoded-gradient fallback
and selects rank 1 only when factor bytes plus residual bytes are smaller.

The decoder receives `(rows + columns)` FP16 values and recreates the exact
encoder-side predictor. The method is causal, shard-local, bounded-memory, and
linear in shard size. Its three matrix-vector passes are more expensive than
plain quantization but do not require global tensor assembly or a model-sized
learned compressor. Factor overhead becomes negligible for large matrices.

Format version 4 stores the factor extent in each selected tensor descriptor
and retains decode compatibility with versions 1-3. E1 intentionally excludes
D5 sparse outliers so endpoint fidelity can be compared with the strict D2
lineage without a second changed variable.

## Results

The causal-posthoc run and replay completed under
`<experiments>/experiment_003/lossy/ots_rank1_deltaq_e1/causal_posthoc/`.

| Metric | E1 |
| --- | ---: |
| Source bytes | 19,491,431,115 |
| Compressed bytes | 8,905,543,932 |
| Compression ratio | 2.18869x |
| Gradient energy R2 | 0.999988138 |
| Gradient cosine | 0.999994069 |
| Encode time | 90.69 s |
| Decode time | 43.10 s |
| Replay loss MAE | 3.491e-5 |
| Replay loss RMSE | 5.068e-5 |
| Final-weight cosine | 0.999994711 |
| Final-weight relative L2 | 0.00325502 |

Rank 1 was selected for 12,181 of the 25,000 matrix instances and predicted
2.370 billion of 4.870 billion gradient elements. The factors consumed only
24.75 MB. E1 moved 891.5 million values into int8, versus 138.0 million in D2,
while FP16 use fell from 4.715 billion to 3.969 billion values. It improves the
compression ratio 8.3% over D2's 2.020x without D5's sparse exceptions.

## Final-model output agreement

On 131,072 held-out next-token predictions, the replayed model has 99.8466%
top-1 agreement with the normal final model. Reference-to-lossy KL is
5.561e-6, label cross-entropy changes by -5.186e-6, and logit relative L2 is
1.2381%. The complete comparison is in `final_model_outputs.csv`.

## Prediction trace

The decoder-reconstructed `predictions.otsg` contains all 1,000 prediction
steps on the external drive. `prediction_comparison.jsonl` contains per-step
energy-R2, cosine, and relative-L2 values for plotting. Future post-hoc runs now
save both predictions and prediction metrics during their single encoding pass;
E1's trace was extracted from the finished artifact without re-encoding it.

| Per-step prediction statistic | Value |
| --- | ---: |
| Mean / median energy R2 | -0.1758 / -0.2352 |
| Energy R2 5th / 95th percentile | -0.7639 / 0.8761 |
| Steps with positive energy R2 | 259 / 1,000 |
| Mean / median cosine | 0.3790 / 0.3353 |

The whole-model prediction R2 remains negative because non-matrix tensors and
unprofitable matrices use the noisy previous-step fallback. Local rank-1
selection still reduces residual dynamic range enough to improve bytes. As in
D2, aggregate predictor quality is not a sufficient proxy for quantized rate.

## What we learned

Learned matrix structure is real and usable, but rank 1 alone is not enough to
reach D5's rate. E1 is a clean middle point: its 0.326% final-weight difference
is about nine times smaller than D5's, but about 6.8 times larger than D2's
0.0482%. Its loss curve and final-weight direction remain extremely close.

The factor cost is negligible, and a one-iteration solver is sufficient for
candidate generation. The remaining limitations are that the coherent matrix
energy fades during training, and the strict residual codec still stores 3.969
billion values as FP16. A logical E2 would warm-start or temporally predict the
small factors, and add decoder-consistent error feedback or an explicitly
zero-mean residual correction before cautiously revisiting robust int8. That
targets endpoint accumulation while building on the measured structural win.
