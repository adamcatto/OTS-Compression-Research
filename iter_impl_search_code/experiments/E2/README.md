# E2 — temporally tracked rank-1 subspaces

## Hypothesis

E1's fixed all-ones initializer becomes a weak one-iteration estimate as
gradient structure changes later in training. Reusing each matrix shard's last
learned right-singular direction should track the slowly moving subspace,
increase profitable rank-1 selections, and improve rate without changing the
residual fidelity policy.

## Why this follows E1

E1 established that rank-1 matrix structure is useful but selected it for only
12,181 of 25,000 matrix instances. Its factors cost just 24.75 MB, while 3.969
billion values still fell back to FP16. E2 changes only the power iteration's
initializer. The strict 1e-4 per-block error gate, factor encoding, residual
codec, previous-decoded-gradient fallback, and one-iteration compute budget are
unchanged.

## Method and scaling properties

The online encoder retains the normalized right factor learned for each matrix
shard and uses it to initialize the next step's single power iteration. A new
or shape-changed matrix starts from E1's fixed vector. State is updated even
when the rank-1 candidate is not selected, because the factor may become useful
at a later step.

The state is encoder-only and never needed to decode: selected FP16 factors are
still transmitted in full, and the decoder recreates the exact predictor from
them. State is proportional to the sum of matrix-shard column counts rather
than parameter count. Work remains three matrix-vector passes per shard, and
each tensor still falls back to its ordinary E1 encoding unless factors plus
residual are smaller.

For this model the tracker stores 11,008 FP32 values, or 44,032 bytes, compared
with 4.87 million gradient values per step. At larger scale this remains
shard-local; the relevant overhead is `4 * sum(local matrix columns)` bytes per
worker rather than four bytes per model parameter.

## Reproduction

`algorithm.py` constructs the exact codec configuration. The controlled run is:

```bash
python -m ots_compression.compression.lossy_cli \
  --experiment experiment_003 \
  --prediction rank1_tensor \
  --rank1-power-iterations 1 \
  --rank1-warm-start \
  --access-label causal-posthoc \
  --block-size 16384 \
  --relative-squared-error 1e-4
```

Large artifacts live under
`<experiments>/experiment_003/lossy/ots_rank1_tracking_e2/causal_posthoc/`.
`results.csv` contains one row per step, including codec-only online latency,
compression ratio, throughput, and fidelity measurements.

## Results

The single causal-posthoc run and replay completed on `experiment_003`.

| Metric | E2 | E1 |
| --- | ---: | ---: |
| Compression ratio | 2.24667x | 2.18869x |
| Compressed bytes | 8,675,703,876 | 8,905,543,932 |
| Gradient energy R2 | 0.999985594 | 0.999988138 |
| Gradient cosine | 0.999992797 | 0.999994069 |
| Rank-1 matrix selections | 15,740 | 12,181 |
| Rank-1 predicted elements | 3.056B | 2.370B |
| Int8 elements | 1.129B | 0.892B |
| Decode time | 43.27 s | 43.10 s |
| Replay loss MAE | 4.374e-5 | 3.491e-5 |
| Final-weight cosine | 0.999993156 | 0.999994711 |
| Final-weight relative L2 | 0.00369975 | 0.00325502 |

The outer post-hoc encode timer was 145.30 s because it includes saving the
18 GB prediction trace and full-gradient instrumentation. The codec-only online
times stored before diagnostic work sum to 81.38 s, or 81.38 ms/step.

| Per-step measurement | Mean | Median | p05 | p95 |
| --- | ---: | ---: | ---: | ---: |
| Compression ratio | 2.2786x | 2.1690x | 2.0740x | 3.0905x |
| Compression latency | 81.38 ms | 76.57 ms | 73.39 ms | 101.37 ms |
| Throughput | 234.44 MiB/s | 242.64 MiB/s | 183.28 MiB/s | 253.17 MiB/s |
| Prediction energy R2 | -0.0735 mean | -0.1021 median | -0.6680 | 0.8783 |

All 1,000 actual rows are committed in `results.csv`; this table is not an
aggregate reconstruction. The external directory retains the compressed and
decoded traces, predictions, raw JSONL telemetry, replay loss curve, and
replayed final weights.

## What we learned

Warm-started subspace tracking validates the rate hypothesis. It increases
profitable structural selection by 29.2%, reduces the artifact by 229.8 MB,
and improves E1's compression ratio by 2.65%. Whole-step prediction also
improves: 361 steps have positive prediction R2, versus 259 for E1.

It does not solve endpoint accumulation. Final-weight relative L2 rises from
0.326% to 0.370%, and the replay loss error increases slightly. E2 is therefore
a useful structural rate improvement, not a fidelity improvement. E3 should
retain warm tracking but change only error behavior—preferably a bounded,
shard-local error-feedback or bias-correction mechanism—and must measure its
state cost explicitly for trillion-parameter scaling.
