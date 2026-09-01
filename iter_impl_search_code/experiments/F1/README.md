# F1 — randomized-Hadamard residual mixing

F1 starts the SPECTRA lineage described in `../../F_LINE_RESEARCH.md`. The
results below come from one full run; the completed trace was not re-encoded.

## Hypothesis

D5 showed that rare residual outliers prevent otherwise useful blocks from
using int8. A seeded orthogonal transform should spread that energy across a
block, allowing strict uniform int8 quantization to meet the unchanged 1e-4
relative-SSE gate without sparse exceptions.

## Method

For each full power-of-two gradient block, F1 applies a deterministic random
sign vector and normalized Walsh-Hadamard transform. It quantizes the rotated
residual with the existing symmetric int8 candidate, applies the inverse
transform locally, and accepts it only when the original-coordinate block
passes the fidelity gate. Failed candidates use the unchanged FP16/FP32
fallback. Partial non-power-of-two blocks skip the transform.

The transform seed is part of the codec configuration, not the payload. The
method is causal, stateless, shard-local, and online-capable. Its prototype is
`O(n log n)` and uses ordinary PyTorch operations; a production implementation
would require a fused GPU Hadamard kernel.

The technical premise comes from randomized-rotation quantizers rather than a
claim that gradients themselves are Hadamard-sparse. EDEN uses a randomized
Hadamard transform as a fast, parallelizable surrogate for dense random
rotation in distributed mean estimation
([Vargaftik et al., 2022](https://proceedings.mlr.press/v162/vargaftik22a.html)).
QUIC-FL analyzes the same transform for quick unbiased compression
([Ben Basat et al., 2022](https://arxiv.org/abs/2205.13341)), and TurboQuant
connects energy-spreading rotations and scalar quantization to near-optimal
online vector rate-distortion
([Zandieh et al., 2025](https://arxiv.org/abs/2504.19874)). F1 borrows only the
energy-spreading mechanism: it retains DeltaQ's explicit per-block error test
and fallback rather than importing those papers' distributed-mean objectives.

For a block `x`, F1 forms `y = H D x`, where `D` is a seeded random-sign
diagonal and `H` is the normalized Walsh-Hadamard matrix. Orthogonality gives
`||x - D H Q(y)||_2 = ||y - Q(y)||_2`, so the error test is still meaningful in
the original coordinates. Random signs turn each output into a signed sum of
all block coordinates; concentration reduces the maximum-to-RMS ratio that
made D3's scalar int8 candidate fail. The transform does not improve the
information-theoretic optimum—it makes a cheap max-scaled scalar quantizer
behave more like a vector quantizer on heavy-tailed inputs.

F1 uses the zero predictor intentionally. D2's previous-gradient predictor was
globally poor (mean prediction R2 `-0.4719`) but occasionally useful in large
blocks. Combining it with the transform in this first test would make a rate
change ambiguous. Using the current gradient as the residual isolates whether
orthogonal mixing resolves D5's measured outlier bottleneck. A later global
rate allocator can choose zero, temporal, and structural candidates together.

## Reproduction

```bash
python -m ots_compression.compression.lossy_cli \
  --experiment experiment_003 \
  --prediction zero \
  --block-transform randomized_hadamard \
  --access-label causal-posthoc
```

Large artifacts live under
`<experiments>/experiment_003/lossy/ots_rht_deltaq_f1/causal_posthoc/`.

## Results

The causal-posthoc run and replay completed under
`<experiments>/experiment_003/lossy/ots_rht_deltaq_f1/causal_posthoc/`.

| Compression measurement | F1 | D5 | E2 |
| --- | ---: | ---: | ---: |
| Compressed bytes | 5,654,329,196 | 6,298,265,295 | 8,675,703,876 |
| Compression ratio | 3.44717x | 3.09473x | 2.24667x |
| Gradient energy R2 | 0.999930143 | 0.999952017 | 0.999985594 |
| Gradient cosine | 0.999965073 | 0.999976009 | 0.999992797 |
| Transformed-int8 elements | 4.101B | — | — |
| All int8 elements | 4.104B | 3.552B sparse-mode | 1.129B |
| FP16 elements | 0.763B | 1.170B | 3.731B |
| FP32 elements | 3.15M | — | — |

F1 reduces D5's artifact by 643.9 MB (10.2%) and improves its compression
ratio by 11.4% without exceptions. It improves E2's ratio by 53.4%. Thus the
rate hypothesis is accepted: orthogonal mixing directly resolves much of the
max-scaled quantizer bottleneck.

The cost is substantial in the unfused prototype. Outer encode time was
311.12 s, while the per-step codec timings in `results.csv` average 251.75 ms
(median 237.70 ms, p95 323.04 ms). Throughput averages 74.78 MiB/s. Decode took
153.87 s. D5 encoded in 104.11 s, so F1's Python transform is roughly 2.4x
slower before any production kernel work.

| Per-step measurement | Mean | Median | p05 | p95 |
| --- | ---: | ---: | ---: | ---: |
| Compression ratio | 3.4469x | 3.4509x | 3.3213x | 3.5498x |
| Compression latency | 251.75 ms | 237.70 ms | 230.73 ms | 323.04 ms |
| Throughput | 74.78 MiB/s | 78.16 MiB/s | 57.51 MiB/s | 80.52 MiB/s |
| Reconstruction R2 | 0.99992996 | 0.99992791 | 0.99992091 | 0.99994990 |

All 1,000 rows are committed in `results.csv`. The external directory retains
the compressed and decoded traces, the required per-step prediction trace,
raw JSONL telemetry, replay curve, and model weights.

## Replay and final-model outputs

| Endpoint measurement | F1 | D5 |
| --- | ---: | ---: |
| Replay loss MAE | 5.754e-4 | 5.649e-5 |
| Replay loss RMSE | 7.058e-4 | 1.234e-4 |
| Maximum loss difference | 2.863e-3 | — |
| Final-weight cosine | 0.997588650 | 0.999595629 |
| Final-weight relative L2 | 0.0695681 | 0.0286023 |
| Label CE delta | +1.098e-3 | +1.434e-4 |
| Reference-to-lossy KL | 8.784e-4 | 5.062e-5 |
| Jensen-Shannon divergence | 2.884e-4 | 1.076e-5 |
| Top-1 agreement | 99.7032% | 99.7200% |
| Logit relative L2 | 28.2482% | 11.7954% |

The standalone endpoint hypothesis is rejected. F1 passes the requested
gradient-R2 gate by a wide margin and has an attractive loss curve in absolute
terms, but its final weights and predictive distribution move much farther
than the strict D/E lineage. The output metrics use the same 32 held-out
validation batches and 131,072 predictions as the earlier algorithms; the
complete row is in `final_model_outputs.csv`.

## Signed-error diagnosis

`trajectory_error.json` was computed from the already saved exact and decoded
traces for F1. The same analysis was applied to D5 without rerunning either
codec.

| Accumulation measurement | F1 | D5 |
| --- | ---: | ---: |
| Scalar error mean | -5.36e-11 | +3.80e-11 |
| Scalar error RMS | 3.396e-6 | 2.815e-6 |
| L2 of step-summed error | 0.23667 | 0.20470 |
| Step-summed error / step-summed gradient L2 | 0.2760% | 0.2387% |
| Cumulative error RMS per parameter | 1.072e-4 | 9.276e-5 |

F1's global signed mean is effectively zero, and its step-summed error L2 is
only 15.6% above D5's. Those raw measures do not explain a 2.43x larger final
weight error. The technically plausible inference is that F1 distributes error
into coordinates/tensors to which AdamW's moment normalization is more
sensitive. This is an inference from the mismatch, not a direct measurement of
update Jacobians. It agrees with error-feedback work showing that biased or
state-coupled compression can fail despite good local distortion
([Karimireddy et al., 2019](https://proceedings.mlr.press/v97/karimireddy19a.html))
and with 1-bit Adam's finding that nonlinear adaptive state requires special
treatment
([Tang et al., 2021](https://proceedings.mlr.press/v139/tang21a.html)).

## What we learned and next decision

Keep randomized-Hadamard mixing as a candidate representation, not as the next
mainline codec by itself. It is the largest clean rate gain so far and avoids
D5's sparse index/value overhead. Its dense nearest-rounded errors are not safe
for AdamW replay under a uniform raw-gradient block gate.

F2 should therefore add a decoder-synchronized coarse sensitivity state—one
EMA/RMS statistic per large block—and choose among transformed int8 and higher
precision candidates using a global Lagrangian constrained by both raw gradient
SSE and an optimizer-preconditioned error proxy. APOLLO provides independent
evidence that channel/tensor-level gradient scaling can approximate much finer
Adam adaptation with low-rank/coarse state
([Zhu et al., 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf)).
Do not loosen the R2 budget yet. First test whether allocating F1's existing
error away from sensitive blocks improves endpoint fidelity at similar bytes.
