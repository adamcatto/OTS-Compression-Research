# F2 — optimizer-aware global rate-distortion allocation

F2 is the second SPECTRA iteration and a direct response to F1's endpoint
failure. This writeup records one designated run on `experiment_003`; it is not
a sweep and the completed trace is encoded once.

## Hypothesis

F1's randomized Hadamard transform improved rate from D5's 3.095x to 3.447x,
but its final-weight relative L2 error was 6.96% and reference-to-lossy output
KL was `8.78e-4`. Its scalar error mean was nearly zero and its raw cumulative
error was only 15.6% larger than D5's, despite 2.43x larger final-weight error.
The next falsifiable explanation is therefore not global bias: uniform raw-SSE
allocation puts quantization noise in blocks where AdamW amplifies it.

F2 tests whether moving the *same* `1e-4` per-step raw error allowance away
from optimizer-sensitive blocks improves replay and final-model agreement at a
similar byte rate. It does not loosen F1's fidelity target.

## Method

For every block in the current step, F2 constructs a small independently
decodable frontier:

1. F1's seeded randomized-Hadamard int8 representation;
2. direct FP16;
3. exact FP32.

Let candidate `c` for block `i` have bytes `b_ic`, raw squared error `d_ic`,
and coarse optimizer-weighted error `a_ic = s_i d_ic`. F2 selects all block
modes jointly using the separable Lagrangian

```text
minimize  sum_i (b_i,c + lambda d_i,c + mu a_i,c)
subject to
  sum_i d_i,c <= 1e-4 sum_i ||g_i||^2
  sum_i a_i,c <= 1e-4 sum_i s_i ||g_i||^2.
```

For fixed multipliers, each block makes a constant-time choice. The encoder
updates the two multipliers multiplicatively for 40 iterations, retains the
lowest-rate feasible discrete solution, and performs a final
feasibility-preserving local byte cleanup. This couples decisions across every
tensor in a step without buffering a future gradient.

The sensitivity is one scalar per 16,384-value block. From previously decoded
gradients only, encoder and decoder update

```text
vbar_i,t = beta2 vbar_i,t-1 + (1 - beta2) mean(ghat_i,t^2)
s_i,t+1 = 1 / (sqrt(vbar_i,t) + epsilon)^2.
```

The experiment uses `beta2=0.95`, exactly matching its saved AdamW replay
contract, and `epsilon=1e-8`. Step zero uses uniform weight because there is no
decoded history. This is a diagonal first-order proxy for squared Adam update
error, not a claim to reproduce coordinatewise Adam state. Persistent codec
state is one scalar per block—about one 64-bit value per 16,384 parameters in
the prototype—and each shard can allocate independently.

The approach is motivated by 1-bit Adam's demonstration that compression must
respect adaptive optimizer state rather than treating all gradient errors as
equivalent ([Tang et al., 2021](https://proceedings.mlr.press/v139/tang21a.html)).
APOLLO independently finds that channel/tensor-level scaling can approximate
much finer Adam adaptation, supporting a coarse sensitivity proxy rather than
a second parameter-sized optimizer
([Zhu et al., 2025](https://proceedings.mlsys.org/paper_files/paper/2025/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf)).
The global candidate allocation is a conventional Lagrangian
rate-distortion construction; the randomized rotation retained from F1 follows
the energy-spreading mechanism analyzed by
[EDEN](https://proceedings.mlr.press/v162/vargaftik22a.html) and
[QUIC-FL](https://arxiv.org/abs/2205.13341).

## Reproduction

```bash
PYTHONPATH=src \
OTS_EXPERIMENTS_DIR=/Volumes/agocdrive/gradient-research/experiments \
python -m ots_compression.compression.lossy_cli \
  --experiment experiment_003 \
  --prediction zero \
  --block-transform randomized_hadamard \
  --allocation optimizer_aware_global \
  --preconditioned-relative-squared-error 1e-4 \
  --sensitivity-beta2 0.95 \
  --sensitivity-epsilon 1e-8 \
  --allocator-iterations 40 \
  --access-label causal-posthoc
```

Large artifacts live under
`<experiments>/experiment_003/lossy/ots_rht_adam_allocator_f2/causal_posthoc/`.
The Git bundle retains the exact construction, this writeup, all per-step
online rate/latency/fidelity rows, and the held-out final-model output row.

## Results

The designated causal-posthoc run, decode, replay, and held-out evaluation all
completed. No stage was rerun.

| Compression measurement | F2 | F1 |
| --- | ---: | ---: |
| Compressed bytes | 4,881,928,084 | 5,654,329,196 |
| Compression ratio | 3.992568x | 3.447170x |
| Gradient energy R2 | 0.999915968 | 0.999930143 |
| Gradient cosine | 0.999957987 | 0.999965073 |
| RHT-int8 elements | 4.865710B | 4.101B |
| All int8 elements | 4.870318B | 4.104B |
| FP16 elements | 82,176 | 0.763B |
| FP32 elements | 0 | 3.15M |

F2 reduces the F1 artifact by 772.4 MB (13.7%) and improves its compression
ratio by 15.8%. It sends 99.9983% of values through int8. The complete trace
still clears the unchanged fidelity target by a wide margin, although its raw
error is 20.3% larger than F1's.

The global constraints held on every emitted step. Raw allocator relative SSE
averaged `8.391e-5` and reached at most `9.962e-5`; the optimizer proxy averaged
`8.819e-5` and reached at most `9.083e-5`. The latter observation is the key
failure diagnosis: the sensitivity constraint was never close to its `1e-4`
limit, so it did not trade substantial precision toward sensitive blocks.

| Per-step measurement | Mean | Median | p05 | p95 |
| --- | ---: | ---: | ---: | ---: |
| Compression ratio | 3.99056x | 3.99062x | 3.99062x | 3.99062x |
| Compression latency | 345.01 ms | 337.05 ms | 295.12 ms | 424.03 ms |
| Throughput | 54.69 MiB/s | 55.12 MiB/s | 43.82 MiB/s | 62.95 MiB/s |
| Reconstruction R2 | 0.99991609 | 0.99991552 | 0.99990907 | 0.99992401 |
| Optimizer-proxy relative SSE | 8.819e-5 | 8.817e-5 | 8.683e-5 | 8.964e-5 |

All 1,000 rows, including both allocator constraints and sensitivity extrema,
are committed in `results.csv`. Outer encode time was 406.99 s and decode time
was 160.91 s. The per-step codec mean is 37.0% slower than F1's 251.75 ms
prototype because F2 constructs and scores three candidates per block in
addition to running the transform. The algorithm is linear and shard-local,
but these Python timings reinforce the need for fused candidate generation and
allocation in a production implementation.

## Replay and final-model outputs

| Endpoint measurement | F2 | F1 | D5 |
| --- | ---: | ---: | ---: |
| Replay loss MAE | 5.967e-4 | 5.754e-4 | 5.649e-5 |
| Replay loss RMSE | 7.285e-4 | 7.058e-4 | 1.234e-4 |
| Maximum loss difference | 2.974e-3 | 2.863e-3 | — |
| Final-weight cosine | 0.997551790 | 0.997588650 | 0.999595629 |
| Final-weight relative L2 | 0.0700972 | 0.0695681 | 0.0286023 |
| Label CE delta | +1.110e-3 | +1.098e-3 | +1.434e-4 |
| Reference-to-lossy KL | 9.080e-4 | 8.784e-4 | 5.062e-5 |
| Jensen-Shannon divergence | 2.982e-4 | 2.884e-4 | 1.076e-5 |
| Top-1 agreement | 99.6552% | 99.7032% | 99.7200% |
| Logit relative L2 | 28.4749% | 28.2482% | 11.7954% |

The same 32 validation batches and 131,072 predictions were used. F2 is
slightly worse than F1 on every main endpoint measure: final-weight relative
L2 grows 0.76%, reference-to-lossy KL grows 3.37%, and JS grows 3.39%. The
standalone endpoint hypothesis is rejected. The raw teacher-to-lossy cross
entropy is `2.217237116`; as before, KL/JS are more discriminating because they
remove the reference distribution's intrinsic entropy. The complete output
row is in `final_model_outputs.csv`, and the external replay directory retains
both final weights and the 1,000-point loss curve.

## Signed-error diagnosis

The saved exact and decoded traces were compared without re-encoding.

| Accumulation measurement | F2 | F1 | D5 |
| --- | ---: | ---: | ---: |
| Scalar error mean | +1.39e-11 | -5.36e-11 | +3.80e-11 |
| Scalar error RMS | 3.725e-6 | 3.396e-6 | 2.815e-6 |
| L2 of step-summed error | 0.25938 | 0.23667 | 0.20470 |
| Step-summed error / gradient L2 | 0.3025% | 0.2760% | 0.2387% |
| Cumulative error RMS per parameter | 1.175e-4 | 1.072e-4 | 9.276e-5 |

F2 still has essentially zero global scalar bias, but its cumulative error L2
is 9.6% larger than F1's. The largest accumulated tensor error is in the token
embedding, followed by early-layer attention matrices. This is consistent with
the endpoint regression and shows that the extra int8 rate was not free.

## What we learned and next decision

Global allocation itself is valuable for rate: F1's independent per-block gate
was highly conservative, and a per-step gate safely moved another 0.766B values
to int8 by the requested raw metric. The optimizer-aware part did not receive a
meaningful test at this threshold because transformed int8 already satisfied
the weighted constraint on nearly every step. A proxy can be theoretically
reasonable and still be operationally inert when its budget is calibrated in
the wrong units.

Do not proceed directly to a larger structural model while retaining this
inactive constraint. The immediate next controlled iteration should rate-match
F1: carry a byte reservoir across steps and tighten/adapt the optimizer-proxy
budget until F2 spends approximately F1's 5.654 GB, forcing the allocator to
place FP16 capacity according to sensitivity. That isolates *placement* from
the amount of error. If rate-matched sensitivity allocation still fails to
improve endpoint fidelity, the block-scalar diagonal proxy is rejected and F3
should move to decoder-synchronized tensor subspaces or F4's sketch-space
signed error debt rather than retuning another scalar threshold.

## Decision rule

This criterion was not met: rate improved, but endpoint drift did not. The next
iteration above is therefore a rate-matched falsification of the allocation
signal, not another relaxation of raw R2.
