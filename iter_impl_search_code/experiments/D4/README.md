# D4 — adaptive per-block predictor selection

The exact runnable construction and result table are colocated in this bundle.

## Hypothesis

A per-block selector between zero and previous-decoded prediction will retain
D2's int8 coverage on large tensors and capture D3's wins on small/partial
blocks. Because both candidates are decoder-available, the selector needs no
side model or learned parameters.

## Why this follows D3

D3 rejected zero as a global predictor, but it won on more individual blocks.
Element-aware telemetry showed that D2's previous-gradient predictor was much
better on large blocks. Choosing one predictor globally therefore discards
useful local structure.

## Method and format

For each 16,384-element block, the encoder evaluates both decoder-known
predictions under the unchanged 1e-4 squared-error gate. It chooses the
candidate with the narrowest valid payload (int8, then FP16, then FP32); ties
choose lower reconstruction error. The existing one-byte block mode stores a
predictor bit, so selection adds no bytes. OTS-DeltaQ format version 2 encodes
this bit while the decoder retains version-1 compatibility.

The implementation remains shard-local, bounded-memory, and available in the
training-time online writer. The controlled first measurement reuses
`experiment_003` and is labeled `causal_posthoc`.

## Results

On `experiment_003`, D4 produced a 9,630,192,199-byte artifact from the
19,491,431,115-byte source: 2.02399x compression. This is 18,448,067 bytes
(0.19%) smaller than D2 and 128,604,213 bytes smaller than D3. Encoding took
88.62 s, 81% longer than D2 because both candidates are evaluated; decoding
took 43.08 s.

| Compression/replay measurement | D4 adaptive | D2 previous | D3 zero |
| --- | ---: | ---: | ---: |
| Compression ratio | 2.02399x | 2.02012x | 1.99732x |
| Gradient energy-R2 | 0.9999952482 | 0.9999961290 | 0.9999979210 |
| Final-weight cosine | 0.9999998737 | 0.9999998838 | 0.9999999670 |
| Final-weight relative L2 | 0.00050254 | 0.00048205 | 0.00025706 |
| Replay loss MAE | 0.00001835 | 0.00001698 | 0.00000527 |
| Replay loss RMSE | 0.00002567 | 0.00002456 | 0.00000714 |
| Maximum replay loss difference | 0.00010633 | 0.00010538 | 0.00002861 |

D4 selects zero prediction for 4,061,833,472 elements (83.4%) and the previous
decoded gradient for 808,566,528 elements (16.6%). It places 156,379,392
elements in int8, versus 137,999,360 for D2 and 29,873,408 for D3. Even so,
4,696,866,560 elements (96.4%) remain FP16.

## Final-model output agreement

On 131,072 held-out next-token predictions, the replayed model has 99.9352%
top-1 agreement with the normal final model. Reference-to-lossy KL is
6.064e-7, label cross-entropy changes by +2.459e-5, and logit relative L2 is
0.0403%. The complete comparison is in `final_model_outputs.csv`.

## What we learned

The local selector does combine D2 and D3's useful cases and slightly improves
rate without selector bytes. It is not a compelling mainline trade: the 0.19%
artifact improvement costs 81% more encode time and produces slightly worse
optimizer replay. Although ties choose the locally lower reconstruction error,
early choices change the future decoded predictor state, so this local rule
does not guarantee a globally lower trajectory error than D2.

## Next decision

A learned temporal predictor is not yet justified. D2-D4 show noisy minibatch
gradients, modest gains from predictor choice, and substantial extra compute.
The dominant bottleneck is instead the max-scaled quantizer: 96.4% of elements
still fall back to FP16. D5 should preserve a small sparse set of block outliers
at FP16/FP32 and int8-quantize the inlier residual with a robust scale. This
directly attacks the measured failure mode, remains linear/shard-local, and is
more likely to create a material rate gain. Predictor learning can be revisited
after establishing that stronger quantization control.

Artifacts and replay results are under
`<experiments>/experiment_003/lossy/ots_deltaq_adaptive_predictor_v1/causal_posthoc/`.
