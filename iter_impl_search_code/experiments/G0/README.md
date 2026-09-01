# G0 — aggressive structural feasibility audit

G0 is an analysis-only opening experiment for the G lineage. It does not
encode, decode, or replay a new gradient artifact. Its purpose is to test
whether the completed `experiment_003` trace contains enough structure to make
the hard 100x target (`0.32` total bits per FP32 gradient value) plausible
before implementing another full codec.

## Hypothesis

A tensor generator can reach 100x only if nearly all gradient energy is
available from decoder-synchronized state or from an exceptionally compact
current-step structure. The remaining innovation must then have entropy below
the few tenths of a bit/value left after codes, metadata, state refreshes, and
keyframes. If even optimistic structural upper bounds miss that condition, a
new precision allocator or a larger rank is not a credible next experiment.

## Method

`algorithm.py` indexes the existing 18 GB exact trace without copying it and
decodes only steps `0, 1, 99, 100, 499, 500, 998, 999`. Measurements are
reported for the 26 matrix-shaped tensors at target steps `0, 1, 100, 500,
999`:

- current-tensor singular-value energy curves;
- immediate-previous-gradient prediction and an oracle fitted scalar;
- one-sided and two-sided projection into the preceding exact tensor's SVD
  bases, an optimistic proxy for decoder-synchronized subspaces;
- preceding-step bases shared across homologous transformer layers;
- balanced Kronecker rearrangement spectra;
- architecture-aware tensor-train SVD at ranks 1, 2, 4, 8, and 16;
- best-coefficient and fixed-low-frequency 2-D Fourier energy;
- an iid-Gaussian residual rate calculation; and
- a tuned uniform scalar residual quantizer's empirical zero-order entropy.

The current-step SVD, oracle scalar, best Fourier support, and exact
preceding-step bases are deliberately favorable upper bounds, not online codec
results. The Gaussian calculation is a conditional source-model bound, not a
distribution-free impossibility theorem. The scalar entropy estimate omits
headers and finite-code penalties. These optimistic assumptions make a miss
more informative than a win.

## Hard rate ledger

The trace has 4,870,400 values per step, so 100x permits exactly 194,816 bytes
per step before any allowance for metadata. The 20,480 non-matrix values alone
would consume 81,920 bytes in FP32 or 40,960 bytes in FP16. Storing them that
way would leave the matrix tensors only 0.186 or 0.254 bits/value,
respectively. Small tensors therefore also need prediction, procedural
regeneration, or a globally allocated entropy code.

For the 26 matrix-shaped tensors, a local rank-8 FP16 basis snapshot is
409,616 bytes, a homologous-layer shared snapshot plus the two singleton
embedding bases is 81,936 bytes, and all
rank-8 FP16 cores total 3,328 bytes per step (`0.00547` bits per full-gradient
value). A local snapshot interval of at least 22 steps is needed merely to keep
its amortized bytes below 10% of the total rate budget. Deterministic basis
evolution can avoid frequent refresh payloads, but then cross-platform
numerical synchronization and random access become explicit format problems.

For an iid Gaussian residual with energy fraction `q`, ideal squared-error rate
at global relative SSE `0.01` is

```text
R = max(0, 0.5 log2(q / 0.01)) bits/original-value.
```

Even with a free structural predictor, `R <= 0.32` requires it to capture at
least 98.4417% of the gradient energy. Real headers and structural codes push
the required capture higher.

## Results

| Optimistic/causal diagnostic | Value-weighted result |
| --- | ---: |
| Current rank-1 energy | 65.58% |
| Current rank-8 energy | 94.15% |
| Current rank-32 energy | 99.36% |
| Causal previous-basis rank-8, left only | 55.35% |
| Causal previous-basis rank-8, right only | 82.06% |
| Causal previous-basis rank-8, two-sided | 47.93% |
| Shared-across-layers causal rank-8, two-sided | 32.82% |
| Kronecker rank-8 energy | 16.14% |
| Architecture-aware TT rank-8 energy | 55.11% |
| TT rank-8 FP16 factors | 0.2585 bits/value |
| Best 1% Fourier coefficients, indices free | 11.25% |
| Fixed low-frequency 1% Fourier coefficients | 1.19% |
| Oracle current-rank-8 Gaussian residual rate | 0.794 bits/value |
| Causal rank-8 Gaussian residual rate | 2.443 bits/value |
| Oracle rank-8 scalar residual entropy | 0.925 bits/value |
| Causal rank-8 scalar residual entropy | 2.534 bits/value |

The trace is extremely low rank at initialization and becomes materially less
so later:

| Step | Current rank-8 energy | Current rank-32 energy | Causal rank-8 two-sided energy |
| ---: | ---: | ---: | ---: |
| 0 | 99.887% | 99.990% | unavailable |
| 1 | 99.880% | 99.989% | 97.332% |
| 100 | 97.832% | 99.748% | 52.762% |
| 500 | 85.474% | 98.697% | 20.624% |
| 999 | 87.664% | 98.370% | 20.999% |

At step 999, even a free current rank-32 predictor leaves a Gaussian-model
residual rate of 0.3526 bits/value, already above the complete budget. Sending
the FP16 rank-32 factors would add roughly 2.705 bits/value. Across the five
sampled steps, the best optimistic combination among tested FP16 SVD factors
and Gaussian residual coding is approximately 0.20 bits/value at initialization,
1.06 at step 100, 2.37 at step 500, and 2.37 at step 999. The late-trace result
is about 7.4 times the hard budget before headers.

Tensorization does not rescue the tested axis order. TT rank 8 spends 0.2585
bits/value on FP16 factors but captures only 55.1% energy; TT rank 16 captures
78.2% over all samples while its factors already cost about 0.774 bits/value.
The weak fixed Fourier result is direct evidence against a standalone smooth
convolution/Hyena-style coordinate generator. A learned nonlinear generator
could still outperform Fourier truncation, but it must demonstrate that gain
with its complete model/update/code rate charged.

## Decision

The gradient-only, standalone low-rank/Kronecker/TT/convolution hypothesis is
rejected at 100x on this audit. The result does **not** establish a universal
impossibility theorem: residuals may be sufficiently non-Gaussian, factor
innovations may compress better than full factors, and training-transcript side
information can change the source entirely.

The smallest decisive follow-up is G1, a no-replay late-step rate-frontier
experiment. It should encode current basis innovations against the preceding
decoder basis, a quantized core, sparse exceptions, and an entropy-coded bulk
residual under one actual 0.32-bit global ledger. It should test steps 100, 500,
and 999 first. If an oracle mode selector cannot reach global R2 0.99 at the
measured byte rate, the pure gradient-only structural path should stop before a
1,000-step replay. In parallel, a procedural control should quantify the much
stronger side-information case: batch identifiers, RNG, code/model hash, and
deterministic forward/backward regeneration.

## Reproduction

```bash
python3 iter_impl_search_code/experiments/G0/algorithm.py \
  --trace /Volumes/agocdrive/gradient-research/experiments/experiment_003/gradients.otsg \
  --steps 0,1,100,500,999 \
  --output iter_impl_search_code/experiments/G0/results.csv \
  --summary iter_impl_search_code/experiments/G0/summary.json
```

`results.csv` contains all 130 sampled tensor rows and `summary.json` contains
the exact aggregates quoted here. They total less than 250 KB. There is no
`final_model_outputs.csv`, compression latency, or prediction trace because G0
does not create a compressor or replay a model; reporting invented online
metrics would mislabel the analysis.

## Primary references

- Oseledets, “Tensor-Train Decomposition,” SIAM J. Sci. Comput. 2011.
  [Paper](https://doi.org/10.1137/090752286)
- Novikov et al., “Tensorizing Neural Networks,” 2015.
  [Paper](https://arxiv.org/abs/1509.06569)
- Martens and Grosse, “Optimizing Neural Networks with Kronecker-Factored
  Approximate Curvature,” ICML 2015.
  [Paper](https://proceedings.mlr.press/v37/martens15.html)
- Vogels et al., “PowerSGD,” NeurIPS 2019.
  [Paper](https://papers.neurips.cc/paper_files/paper/2019/hash/d9fbed9da256e344c1fa46bb46c34c5f-Abstract.html)
