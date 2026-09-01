# Experiment bundles

Each implemented algorithm has one directory containing its writeup, exact
algorithm construction, and `results.csv`. New runs store one result row per
training step. Runs completed before per-step instrumentation was introduced
retain their aggregate writeup and explicitly mark unavailable step data rather
than being rerun solely to backfill telemetry.

When a replayed final model exists, the bundle also contains
`final_model_outputs.csv`. These measurements evaluate the normal and
lossy-gradient-replayed final models on the same held-out examples. They are
computed from saved weights, so adding them did not rerun training or codec
experiments.

Analysis-only feasibility bundles are labeled explicitly. They retain exact
code and complete sampled rows, but do not invent compression latency or final
model outputs when no codec artifact or replay was created. `G0` is the first
such pre-implementation gate.

## Final-model output comparison

All rows below use the same 32 validation batches (131,072 next-token
predictions). `H(ref, lossy)` is the cross-entropy from the reference model's
probability distribution to the lossy replay's distribution; KL and JS remove
the reference distribution's intrinsic entropy and are therefore more useful
for comparing algorithms.

| ID | Label CE delta | H(ref, lossy) | KL(ref || lossy) | JS | Top-1 agreement | Logit rel. L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D2 | +1.283e-5 | 2.21632965 | 5.125e-7 | 1.281e-7 | 99.9397% | 0.0376% |
| D3 | +4.299e-6 | 2.21632927 | 1.351e-7 | 3.378e-8 | 99.9710% | 0.0203% |
| D4 | +2.459e-5 | 2.21632974 | 6.064e-7 | 1.516e-7 | 99.9352% | 0.0403% |
| D5 | +1.434e-4 | 2.21637976 | 5.062e-5 | 1.076e-5 | 99.7200% | 11.7954% |
| E1 | -5.186e-6 | 2.21633470 | 5.561e-6 | 1.406e-6 | 99.8466% | 1.2381% |
| E2 | +7.269e-6 | 2.21633710 | 7.963e-6 | 2.038e-6 | 99.8520% | 1.0389% |
| F1 | +1.098e-3 | 2.21720750 | 8.784e-4 | 2.884e-4 | 99.7032% | 28.2482% |
| F2 | +1.110e-3 | 2.21723712 | 9.080e-4 | 2.982e-4 | 99.6552% | 28.4749% |
| F2R | +1.043e-3 | 2.21719157 | 8.624e-4 | 2.834e-4 | 99.7002% | 27.8086% |
| G1P | -5.247e-9 | 2.21632914 | 1.648e-12 | 4.120e-13 | 100.0000% | 0.0000436% |

The raw teacher-to-lossy cross-entropy changes little because most of it is the
teacher's entropy. G1P is the closest output distribution, but it is a
procedural side-information control that reruns training against the external
dataset rather than a standalone gradient codec. Among the self-contained
gradient codecs, D3 is the closest output distribution;
F2 is the most divergent despite its two strict global error gates. Rate-matched
F2R recovers a modest portion of F1/F2's drift, showing that placement matters
but the block-scalar Adam proxy is not sufficient by itself.
