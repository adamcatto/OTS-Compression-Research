# OTS iterative compression research protocol

This directory is the durable memory for algorithm work. Before proposing or
running an iteration, read `ALGORITHM_INDEX.md` and every linked result/writeup
for the active lineage. Do not repeat a completed experiment unless the
writeup names the reason (replication, a corrected bug, or a changed metric).

## Loop

1. State one hypothesis and the immediate predecessor it builds on.
2. Implement the smallest testable variant, with a decoder and unit tests.
   A method labeled online must emit and flush each encoded step inside the
   training loop; replaying a completed trace with causal state is reported as
   causal post-hoc evaluation, not an online runtime result.
3. Run it once against the designated trace. Large encoded and decoded files
   live under the external experiment directory, never in git.
4. Measure compressed size, encode/decode cost, gradient energy-R2, cosine
   similarity, replayed optimizer trajectory, final-weight difference, and
   fixed-probe loss curve when their required artifacts exist. On identical
   held-out examples, compare the normal and replayed final models using label
   cross-entropy, teacher-to-lossy cross-entropy, KL in both directions,
   Jensen-Shannon divergence, output agreement, and logit/probability error.
5. Store the experiment's writeup, runnable algorithm definition, aggregate
   result, and per-step result table together under `experiments/<ID>/`, then
   add/update the index.
6. Let the measured failure mode choose the next change. A departure from the
   lineage is allowed only when the writeup explains why the old line is not
   promising.

## Fidelity gates

Gradient energy-R2 is `1 - ||g - ghat||² / ||g||²`; this is the threshold used
for early filtering and is intentionally stronger and less ambiguous than a
correlation-only score. A candidate must clear 0.99 before replay analysis.
It is not a release criterion: AdamW update error, final-weight distance, and
fixed-probe loss are the deciding measurements.

## Result locations

`<experiment>/lossless/` holds exact codec controls. `<experiment>/lossy/`
holds lossy candidates and off-the-shelf controls. Old `baselines/` directories
are legacy results and are not copied merely to change names.

## Git experiment bundle

Every implemented algorithm has one self-contained directory:

```text
experiments/<ID>/
  README.md       hypothesis, method, aggregate result, and interpretation
  algorithm.py    exact runnable algorithm construction used for the run
  results.csv     one row per training step
  final_model_outputs.csv  held-out final-model distribution comparison
```

`results.csv` stores at least the step number, uncompressed and compressed
bytes, compression ratio, online compression latency, and compression
throughput. Prediction and reconstruction metrics are included when available.
`final_model_outputs.csv` is required whenever a replayed final model exists.
Large tensors, compressed artifacts, model weights, and loss curves remain on
the external experiment drive. A legacy run whose per-step latency was not
recorded is marked unavailable rather than rerun only to populate the table.
