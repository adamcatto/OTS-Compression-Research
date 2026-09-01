# D3 — zero-predictor DeltaQ ablation

The exact runnable construction and result table are colocated in this bundle.

## Hypothesis

Replacing D2's previous-decoded-gradient predictor with zero will shrink the
residual dynamic range relative to current-gradient energy on this stochastic
minibatch trace, allowing more blocks to use int8 while preserving the exact
same per-block error gate and replay fidelity.

## Why this follows D2

D2 measured mean prediction energy-R2 of -0.4719 and found that the temporal
predictor beat zero on only 132 of 1,000 steps. It also found that 88.4% of
blocks fell back to FP16. D3 changes only the prediction; block size, quantizer,
fallbacks, and 1e-4 relative squared-error budget remain fixed.

## Method and access regime

The prediction for every tensor is identically zero, so the encoded residual is
the current gradient. This is trivially causal, stateless, shard-local, and
available in the training-time online writer. The first controlled measurement
uses the completed `experiment_003` trace and is therefore labeled
`causal_posthoc`, not runtime-online. Reusing that exact trace isolates the
predictor change and avoids an uninformative repeat of deterministic training.

## Results

On `experiment_003`, the zero predictor produced a 9,758,796,412-byte artifact
from 19,491,431,115 source bytes: 1.99732x compression. It is 110,156,146 bytes
larger than D2 and therefore rejects the rate hypothesis. Encode/decode took
55.03 s and 44.26 s.

| Compression/replay measurement | D3 zero | D2 previous decoded |
| --- | ---: | ---: |
| Compression ratio | 1.99732x | 2.02012x |
| Gradient energy-R2 | 0.9999979210 | 0.9999961290 |
| Gradient cosine | 0.9999989605 | 0.9999980645 |
| Final-weight cosine | 0.9999999670 | 0.9999998838 |
| Final-weight relative L2 | 0.00025706 | 0.00048205 |
| Replay loss MAE | 0.00000527 | 0.00001698 |
| Replay loss RMSE | 0.00000714 | 0.00002456 |
| Maximum replay loss difference | 0.00002861 | 0.00010538 |

The zero predictor improves fidelity because more data remains FP16, not
because it produces a better rate/fidelity frontier.

## Mode allocation correction

D3 initially appeared to have more int8 blocks (44,910 versus D2's 39,137),
despite its larger artifact. Element-aware inspection resolved this:

| Mode | D3 elements | D2 elements |
| --- | ---: | ---: |
| int8 | 29,873,408 (0.61%) | 137,999,360 (2.83%) |
| FP16 | 4,822,323,968 (99.01%) | 4,715,213,824 (96.81%) |
| FP32 | 18,202,624 (0.37%) | 17,186,816 (0.35%) |

The extra D3 int8 blocks belong mostly to small or partial tensor blocks. The
previous-gradient predictor, although poor by whole-step energy-R2, enables
int8 on substantially more elements in large blocks. Whole-step predictor R2
does not by itself predict quantizer rate under a max-scaled block quantizer.

## What we learned and next decision

The zero ablation is useful but not the next mainline codec. A stronger D4 is a
per-block predictor selector: both zero and previous-decoded predictions are
decoder-available, so the encoder can choose the cheaper valid representation
and signal that choice in the block header. This should retain D2's large-block
wins, capture D3's small-block wins, and never select a worse payload except for
the small selector overhead. Evaluate that before low-pass or learned predictors.

Artifacts and replay results are under
`<experiments>/experiment_003/lossy/ots_deltaq_zero_v1/causal_posthoc/`.
