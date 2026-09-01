# G lineage: aggressive structural generation at 0.32 bits/value

This line asks a different question from D through F. Those lineages improved
the numerical representation of an essentially dense gradient stream. G treats
each dense tensor as the output of a compact causal program: analytic factors,
decoder-synchronized state, a shared generator, and a small innovation record.
The hard goal is at least 100x relative to FP32, so the complete archive must
average no more than `0.32` bits per gradient value.

G starts with feasibility evidence rather than a full codec. F2R already shows
that careful placement at 9.27 payload bits/value reaches only 3.447x and still
accumulates 6.84% final-weight relative L2 error. A 29x rate reduction from
F2R cannot plausibly come from another threshold adjustment.

## Required endpoint contract

A full G experiment must report all of the following:

- actual complete bits/value including codes, tensor descriptors, entropy
  tables, decoder/model updates, keyframes, hashes, and amortized state refresh;
- per-step global dense-gradient energy R2, with a gate of at least 0.99;
- online encode latency and decode latency/throughput;
- exact causal access assumptions and all persistent encoder/decoder state;
- replay loss drift, final-weight cosine and relative L2;
- label CE, teacher-to-replay CE, KL in both directions, Jensen-Shannon
  divergence, top-1 agreement, logit/probability error; and
- predictions and prediction differences for every step of a full run.

The final-model top-1 target is at least 0.99 versus the exact-gradient
reference. That is necessary but not sufficient: E2 already achieved 0.99852
agreement at only 2.247x while its final weights moved 0.370%. F2R achieved
0.99700 at 3.447x while final weights moved 6.84%. Loss curves and top-1 labels
can conceal material distribution and trajectory drift.

## Feasibility audit

G0 reads eight selected records from the existing exact `experiment_003`
trace and writes no tensor duplicate. It measures 130 matrix snapshots at
steps `0, 1, 100, 500, 999`. The exact code and complete rows are in
`experiments/G0/`.

### Existing evidence retained from E and F

- E1's broader 50-step sample found fitted lag-1 R2 `0.17596`, AR(2) R2
  `0.21841`, cross-layer R2 `0.05908`, adjacent-coordinate R2 `0.00195`, and
  rank-1 matrix energy `0.59466`.
- E2's warm rank-1 tracker increased structural selections by 29.2% but reached
  only 2.247x and slightly worsened endpoint drift.
- F2R demonstrated that optimizer-sensitive rate placement helps continuous
  endpoints at equal rate, but only by 1–6%; it does not create the missing
  order of magnitude.

No completed E/F experiment was rerun for G0.

### New G0 measurements

| Diagnostic | Value-weighted result | Interpretation |
| --- | ---: | --- |
| Current rank-8 / rank-32 energy | 94.15% / 99.36% | Strong current structure, but the basis is not free |
| Causal prior-basis rank-8 left / right / two-sided | 55.35% / 82.06% / 47.93% | The input/right subspace persists more than the output/left subspace |
| Shared-layer causal rank-8 two-sided | 32.82% | Raw homologous basis sharing hurts |
| Kronecker rank-8 energy | 16.14% | Balanced separability is weak |
| TT rank-8 energy / FP16 rate | 55.11% / 0.2585 bpv | Nearly consumes the budget before residuals |
| Best 1% / fixed-low 1% Fourier energy | 11.25% / 1.19% | Spectral sparsity is weak; smooth convolution is especially weak |
| Oracle / causal rank-8 scalar residual entropy | 0.925 / 2.534 bpv | Both exceed 0.32 before all other costs |

The low-rank structure is nonstationary. Current rank-8 energy falls from
99.89% at initialization to 85.47% at step 500 and 87.66% at step 999.
Current rank-32 remains high, but at step 999 it captures 98.3696%, below the
98.4417% needed for an iid-Gaussian residual to fit in 0.32 bits/value even if
the rank-32 basis and core were free. The actual FP16 factors cost about 2.705
bits/value.

This is evidence against a standalone method, not a theorem against all
structure. The Gaussian residual formula assumes iid Gaussian innovations;
sparse/non-Gaussian exceptions can code differently. The current SVD is also
an encoder-only oracle. Conversely, the audit gives every Fourier coefficient
free support indices and omits headers/finite entropy overhead, so those
measurements are optimistic.

## The rate budget, explicitly

For `N` FP32 values, 100x permits

```text
B_step = 0.32 N bits = 0.04 N bytes.
```

| Model/trace size | FP32 gradient | Entire 100x record |
| ---: | ---: | ---: |
| 4.8704M values (this trace) | 19.482 MB | 194,816 B |
| 1B values | 4 GB | 40 MB |
| 1T values | 4 TB | 40 GB |

Forty gigabytes per trillion-value step is still a major archive: 1,000 steps
would be 40 TB. The ratio is aggressive; the absolute system problem remains.

For a Gaussian innovation at relative distortion `D=0.01`, Shannon's
rate-distortion function gives `0.5 log2(q/D)` bits per original value when the
predictor leaves energy fraction `q`. Spending the full 0.32 bits on that
innovation requires `q <= 0.0155833`, or at least 98.4417% prediction energy.
If codes, keyframes, and metadata consume 0.08 bits/value, the predictor must
instead leave no more than `0.01 * 2^(0.48) = 1.395%` energy.

The small tensors cannot be ignored. This trace has 20,480 non-matrix values
per step. Exact FP32 storage consumes 42.1% of the whole G budget; FP16 consumes
21.0%. A successful global allocator must regenerate or aggressively model
them rather than silently excluding them from the denominator.

### Decoder state and keyframes

On this trace:

| Rank | Local FP16 bases | Homologous-shared plus singleton bases | FP16 cores/step |
| ---: | ---: | ---: | ---: |
| 4 | 204,808 B | 40,968 B | 832 B |
| 8 | 409,616 B | 81,936 B | 3,328 B |
| 16 | 819,232 B | 163,872 B | 13,312 B |
| 32 | 1,638,464 B | 327,744 B | 53,248 B |
| 64 | 3,276,928 B | 655,488 B | 212,992 B |

Rank-64 cores alone exceed the complete per-step budget. Local rank-8 bases
are modest resident state, but a full snapshot costs 2.10 step budgets. To keep
snapshots under 10% of rate, their interval must be at least 22 steps. At a
square width `d=16,384`, rank-8 FP16 left/right state is `32d` bytes per matrix,
about 0.0977% of an FP16 `d x d` matrix. Extrapolated to a 1T-parameter model
with similarly large matrices, that is roughly 2 GB of resident global basis
state: sublinear per matrix dimension, but not free.

SVD/QR updates are not reliably bit-identical across hardware. A real format
must use quantized/fixed-order state updates, record numerical contracts and
state hashes, and emit recovery keyframes. “Both sides run the same PyTorch
SVD” is not a replay guarantee.

## Candidate families, ranked

| Rank | Family | 100x outlook | Evidence-driven role |
| ---: | --- | --- | --- |
| 1 | Procedural minibatch replay / activation-error regeneration | Plausible with side information; high decode compute | Strongest structural exception and an essential control |
| 2 | Causal basis innovation + core + sparse/bulk residual | Unproven; only gradient-only path worth a decisive test | G1 |
| 3 | Shared hyperdecoder for factors/residual statistics | Plausible only if small, slowly updated, and sublinear | Condition/rate model, not a dense hidden copy |
| 4 | One-sided activation/error subspaces and shared dictionaries | More promising than two-sided rank-8 | Component of G1/G2 |
| 5 | Current low-rank/Tucker/TT/Kronecker plus sparse residual | Rejected standalone by G0 | Oracle/control and occasional keyframe mode |
| 6 | INR/parametric convolution/Hyena/SSM tensor generator | Weak fixed-spectrum evidence; learned test deferred | Small residual or factor generator only |
| 7 | Sketches, error debt, and causal keyframes | Cannot reconstruct dense innovation | Endpoint controller around another codec |
| 8 | Diffusion/score residual generator | Poor rate/compute/replay fit standalone | Deterministic entropy-model prior only |

## 1. Procedural and factor reconstruction

For a linear layer with input activations `A` and backpropagated errors
`Delta`, the minibatch gradient is an outer-product accumulation

```text
G_W = Delta^T A.
```

Convolution has the same form after an `im2col`/patch transformation. This is
the most exact within-layer structure available. It also exposes the central
side-information distinction:

- Saving raw `A` and `Delta` is usually not compression. Here each step has
  4,096 token positions; for a `1024 x 256` FFN matrix, FP16 factors with that
  sample dimension would exceed the 262,144-value gradient many times over.
- Saving only batch identifiers, labels, RNG state, initial model, optimizer,
  and executable/version hashes can make the gradient record extremely small
  because the decoder reruns forward/backward. Dropout is zero in this trace,
  making the control unusually favorable.
- Procedural decoding costs approximately another training forward/backward,
  requires dataset/code availability, and must reproduce kernels and floating
  point behavior. It is a training transcript, not a lightweight standalone
  gradient archive.

At billion/trillion scale, the decoder already needs the model and optimizer
contract for endpoint replay; procedural regeneration adds activation memory
and training-scale compute but no model-sized *additional* learned decoder.
This option should be reported separately as “replay with external training
inputs,” never compared as though it were a self-contained gradient codec.
K-FAC's activation/error Kronecker factors support the mechanistic split but do
not imply that an individual minibatch gradient is cheaply recoverable
([Martens and Grosse, 2015](https://proceedings.mlr.press/v37/martens15.html)).

## 2. Decoder-synchronized bases with coded innovation

E1/E2 sent current rank-1 factors. A genuinely different design maintains
quantized bases `U_(t-1), V_(t-1)` at both sides and transmits:

```text
C_t = U_(t-1)^T G_t V_(t-1)
G_base = U_(t-1) C_t V_(t-1)^T
G_t = G_base + basis_innovation_t + sparse_t + bulk_t.
```

The preceding right/input basis is empirically stronger than the left/output
basis: at rank 8 G0 captures 82.06% one-sided right energy versus 55.35% left
and 47.93% two-sided. G1 should therefore test asymmetric ranks and one-sided
cores, not assume square rank `r` is optimal. A right-basis representation
stores `m*r` current coefficients; at FP16 its rate is `16r/n` bits/value,
already 0.5 bits/value for `r=8,n=256`. Coefficients themselves must therefore
be quantized/entropy-coded or produced by a shared decoder.

PowerSGD validates fast low-rank gradient factors, while GROUSE/Oja/PETRELS and
SubTrack-Grad motivate streaming basis updates. G0 adds the missing archive
warning: a previous exact rank-8 two-sided basis captures only 21% at late
steps. Increasing rank without coding basis innovation is not enough.

The scalable state is `O(r sum(m+n))`; projection/update work is `O(rN)` and
decode must write `O(N)` values. At 1T values, even rank 8 means several
trillion multiply-adds per step. Accelerator fusion and sharded local state are
requirements, not optimization notes.

Primary evidence:
[PowerSGD](https://papers.neurips.cc/paper_files/paper/2019/hash/d9fbed9da256e344c1fa46bb46c34c5f-Abstract.html),
[Oja streaming PCA](https://proceedings.mlr.press/v49/jain16.html),
[GROUSE](https://people.eecs.berkeley.edu/~brecht/papers/10.Bal.Now.Rec.GROUSE.pdf),
and [SubTrack-Grad](https://arxiv.org/abs/2502.01586).

## 3. Tensor decompositions and low-rank-plus-sparse structure

Tensor trains use SVDs of successive unfoldings and can represent some
high-dimensional arrays with parameter counts linear in order rather than
element count ([Oseledets, 2011](https://doi.org/10.1137/090752286)). Tensorized
neural weights have achieved very large compression for particular trained
layers ([Novikov et al., 2015](https://arxiv.org/abs/1509.06569)). That does not
transfer automatically to stochastic gradients.

G0 tensorizes attention/FFN dimensions into projection, head, head-width, and
expansion axes. Rank-8 TT factors cost 0.2585 bits/value but capture only 55.1%
energy; rank 16 costs 0.774 bits/value before residuals. Balanced rank-8
Kronecker structure captures 16.1%. The tested decomposition is rejected as a
standalone record, though alternative axis orders and Tucker/CP remain useful
oracle controls.

A low-rank-plus-sparse record can beat a pure low-rank record only if the
innovation energy is concentrated in few entries or blocks. Robust PCA gives
the conceptual decomposition, but its exact recovery assumptions do not hold
by default for dense minibatch noise. G1 must charge sparse indices and show an
actual entropy-coded bulk; reporting only “top-k residual energy” is not a rate
result. [Low-rank and sparse structure pursuit](https://proceedings.mlr.press/v51/gu16.html)
is relevant methodology, not evidence that this trace meets its assumptions.

## 4. Shared dictionaries and hypernetworks

A small shared hypernetwork can condition on layer family, shape factors,
tensor role, normalized step, learning rate, optimizer summaries, basis state,
and a per-step latent. Hypernetworks explicitly support generating non-shared
weights from shared parameters
([Ha et al., 2016](https://arxiv.org/abs/1609.09106)). The archive version must
generate factors, cores, or residual distribution parameters—not naively emit
every dense gradient through a large MLP.

Let the shared decoder have `P` parameters at `s` bytes each and serve `T`
steps of `N` gradients. Its archive cost is `8sP/(NT)` bits/value if sent once.
That can look tiny over a long trace even when `P=Theta(N)`. Resident memory is
the guardrail: a trillion-parameter FP16 decoder is another 2 TB model and is
rejected even if its bytes amortize over 1,000 steps. Require `P=o(N)` globally,
ideally shared across layers with shape-dependent low-rank heads.

Decoder updates create versioning obligations. Every update packet counts
toward 0.32 bits/value; old codes bind to a decoder hash; random access needs
model checkpoints; encoder and decoder train only from mutually decoded data.
Learned Gradient Compression shows that autoencoders can exploit gradient
correlations in distributed training, but its convergence objective and
cross-worker setting do not establish archival R2
([Abrahamyan et al., 2021](https://arxiv.org/abs/2103.08870)).

G0's raw homologous shared bases are worse than tensor-local bases. A shared
dictionary should therefore share *generator parameters and normalized
features*, with small tensor-specific adapters, rather than force raw
coordinates into one basis.

## 5. Implicit neural representations and parametric convolutions

An INR maps coordinates plus conditioning to a gradient value. SIREN and
Fourier-feature networks show how coordinate MLPs represent high-frequency
signals
([Sitzmann et al., 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html),
[Tancik et al., 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html)).
COIN makes the compression contract explicit: overfit an MLP, quantize its
weights, and transmit them
([Dupont et al., 2021](https://openreview.net/pdf?id=yekxhcsVi4)).

For gradients, per-tensor overfitting is noncausal only in compute, not access:
the current tensor is available to its encoder. It is nevertheless a poor
first G implementation:

- G0's fixed low-frequency 1% captures only 1.19% energy, so a smooth/small
  coordinate model has no favorable spectral prior.
- A SIREN capable of fitting noise-like residuals may simply store the tensor
  in weights; complete quantized parameter bytes decide the result.
- Online fitting adds many passes over each tensor and decoder evaluation costs
  at least `O(N)`, commonly `O(Nh)` for width `h`.

The viable version is amortized: one shared coordinate decoder plus a compact
per-step/tensor latent, used on already structured residuals. Its first test
should hold the total latent and amortized update rate below 0.32 and compare
against an equal-byte SVD/sparse control on late tensors.

## 6. Hyena/state-space parameterizations

Hyena interleaves implicitly parameterized long convolutions and data-controlled
gating; S4 parameterizes long linear state-space convolutions efficiently
([Poli et al., 2023](https://proceedings.mlr.press/v202/poli23a.html),
[Gu et al., 2022](https://openreview.net/pdf?id=uYLFoz1vlAC)). A gradient codec
could apply such operators along flattened blocks, matrix rows/columns, or the
time axis, with layer-conditioned filters and a small state.

The analogy has limits. Flattened parameter order is not a semantic token
sequence, and fixed low-frequency FFT support is nearly useless in G0. A
Hyena-inspired model is credible only with axis-aware gating and a complete
latent/update budget. Long temporal convolution is also constrained by E1's
weak lag/AR fits and causality. Complexity is `O(N log B)` or multiple `O(N)`
passes, state is sublinear if filter parameters are shared, and all filter
updates must be decoder-synchronized. Rank below basis-innovation and
procedural controls.

## 7. Diffusion/score-inspired residuals

Score models learn a distribution and reverse a noising process; deterministic
DDIM trajectories show that sampling can be replayable given fixed model,
latent, and numerical contract
([Song et al., 2021a](https://openreview.net/pdf?id=9BnCwiXB0ty),
[Song et al., 2021b](https://openreview.net/pdf?id=St1giarCHLP)). They do not
remove the need to specify which stochastic minibatch residual occurred. If
the conditional score prior is imperfect, the seed/latent and correction code
must carry that innovation entropy.

A dense score network risks being a hidden model copy, and `K` denoising steps
cost at least `O(KN)` tensor work. Determinism also depends on kernels and
precision. G therefore treats diffusion ideas as a learned conditional entropy
model or one/few-step residual refiner around explicit codes, never as evidence
that an untransmitted sample can be reconstructed at R2 0.99.

## 8. Temporal codes, sketches, debt, and keyframes

Raw previous-gradient prediction is weak, but temporal state remains valuable
for factor innovation, rate parameters, error debt, and keyframe cadence.
Count Sketch can hold linear summaries of accumulated error and identify
persistent heavy directions; it cannot reconstruct a dense isotropic residual.
Error feedback improves optimization convergence but changes the training
trajectory objective unless the archive reconstructs every required gradient.

Use three distinct states:

1. **Representation state:** quantized bases/dictionaries that directly predict
   the current tensor.
2. **Entropy state:** causal distributions/scales for arithmetic or ANS coding.
3. **Trajectory state:** sketches of signed and optimizer-preconditioned error
   that force future exceptions or keyframes.

All evolve from decoded data, carry hashes, and have bounded recovery
intervals. FetchSGD supports sketch-space momentum/error accumulation and
error-feedback theory explains why debt matters
([Rothchild et al., 2020](https://proceedings.mlr.press/v119/rothchild20a.html),
[Karimireddy et al., 2019](https://proceedings.mlr.press/v97/karimireddy19a.html)).

## Scaling audit at billion/trillion parameters

| Component | Persistent memory | Encode/decode work | Scaling judgment |
| --- | --- | --- | --- |
| Local rank-r bases | `O(r sum(m+n))` | `O(rN)` | Sublinear state; work can be several full tensor passes |
| TT/Tucker/CP factors | Shape/rank dependent | SVD/ALS encode; `O(rN)`-like decode | Only if ranks stay small; G0 says tested TT ranks do not |
| Shared hyperdecoder | Require `o(N)` params | At least `O(N)`, often `O(Nh)` | Reject per-layer dense heads/model copy |
| Hyena/SSM | Shared filter/state | `O(N log B)` or repeated `O(N)` | Sublinear state, questionable signal match |
| Diffusion residual | Potentially large net | `O(KN)` or worse | Too expensive except tiny/few-step prior |
| Procedural replay | Existing model/optimizer plus activations | Forward + backward | Storage winner, compute-heavy and externally conditioned |
| Sketch debt | Fixed/sketch width | `O(N)` update | Endpoint controller only |

At scale, operate on existing tensor/ZeRO shards and never assemble a whole
model gradient. State snapshots and decoder updates are shard-local records;
global R2 allocation can reduce shard summaries and broadcast only a few
multipliers. Any scheme needing a second dense residual, dense optimizer copy,
or per-layer neural decoder of comparable size to its layer fails the resident
memory criterion even if archive amortization makes its nominal bpv look low.

## Smallest decisive sequence

### G1 — late-step causal structural rate frontier (complete; rejected)

No model replay and no 1,000-step encode yet. On steps 100, 500, and 999:

1. Start from the exact preceding gradient's basis, which is an optimistic
   upper bound on decoded state.
2. Test asymmetric one-sided and two-sided ranks.
3. Encode principal-angle/basis innovation, quantized cores, sparse exceptions,
   and entropy-coded bulk residual into an actual byte stream.
4. Use one global 0.32-bit/value ledger including tensor metadata and an
   amortized keyframe charge.
5. Record encode/decode latency and global energy R2.
6. Compare with oracle current SVD, TT, and best low-rank-plus-sparse frontiers.

Stop if the oracle selector cannot reach R2 0.99. Passing a sampled late-step
frontier only authorizes a short causal prefix; it does not authorize the full
replay.

G1 implemented this screen with actual decodable streams and a global
multiple-choice rate allocator. It widened the candidate set through rank 64,
explicit aligned basis-coordinate innovations, sparse plus entropy-coded bulk
residuals, and paid current-SVD factors. At step 100, causal-only achieved
`R2 = 0.988715` at `0.319202` bpv; adding paid current factors passed with
`R2 = 0.993601` at `0.319281` bpv. At step 500, however, causal-only fell to
`0.960141`, and even the paid-current-factor frontier reached only `0.970510`
at `0.319258` bpv. The gate therefore stopped the run before step 999 and did
not authorize replay. See `experiments/G1/README.md` and `results.csv`.
The G0 TT and Kronecker measurements remained comparison controls rather than
being promoted into byte codecs because their energy capture was substantially
below even the rejected G1 frontier.

### G1P — procedural side-information control

Measure bytes for batch indices, labels if not dataset-derived, RNG state,
initial state/hash, environment/code identifiers, and keyframes. Verify whether
the exact gradients can be regenerated on the same host and quantify one-step
and short-prefix runtime. Report this separately because the decoder has the
dataset and performs training compute.

### G2 — shared factor/residual hyperdecoder (conditional)

Only if G1 identifies compressible factor innovation. G1 did not clear that
condition at the late-step gate, so this branch is currently not authorized.
If reopened with new evidence, fit a shared sublinear
decoder for factor updates or residual distribution parameters. Include all
initial weights and online updates; forbid dense per-layer output heads.

### G3 — short replay and endpoint screen (conditional)

Run enough contiguous steps to expose decoder-state drift and Adam error debt.
Save every prediction. A full 1,000-step replay is justified only after actual
rate, R2, synchronization, and short-prefix trajectory gates pass.

## Pivot rule

G1 reached the rate but not `R2 = 0.99` at step 500 even with optimistic exact
prior bases and a paid-current-SVD comparison frontier. This activates the
pivot: record the incompatibility as a tested gradient-only result and move the
main path to procedural training transcripts or explicitly state the side
information required. Do not relabel 1–3 bits/value as “near 100x,” exclude
small tensors/metadata, or relax the R2 target.

## Selected primary references

1. Oseledets, “Tensor-Train Decomposition.” SIAM J. Sci. Comput. 2011.
   [Paper](https://doi.org/10.1137/090752286)
2. Novikov et al., “Tensorizing Neural Networks.” 2015.
   [Paper](https://arxiv.org/abs/1509.06569)
3. Ha, Dai, and Le, “HyperNetworks.” 2016.
   [Paper](https://arxiv.org/abs/1609.09106)
4. Sitzmann et al., “Implicit Neural Representations with Periodic Activation
   Functions.” NeurIPS 2020.
   [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html)
5. Tancik et al., “Fourier Features Let Networks Learn High Frequency
   Functions in Low Dimensional Domains.” NeurIPS 2020.
   [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html)
6. Poli et al., “Hyena Hierarchy.” ICML 2023.
   [Paper](https://proceedings.mlr.press/v202/poli23a.html)
7. Gu, Goel, and Re, “Efficiently Modeling Long Sequences with Structured
   State Spaces.” ICLR 2022. [Paper](https://openreview.net/pdf?id=uYLFoz1vlAC)
8. Song et al., “Score-Based Generative Modeling through Stochastic
   Differential Equations.” ICLR 2021.
   [Paper](https://openreview.net/pdf?id=9BnCwiXB0ty)
9. Song, Meng, and Ermon, “Denoising Diffusion Implicit Models.” ICLR 2021.
   [Paper](https://openreview.net/pdf?id=St1giarCHLP)
10. Martens and Grosse, “Optimizing Neural Networks with Kronecker-Factored
    Approximate Curvature.” ICML 2015.
    [Paper](https://proceedings.mlr.press/v37/martens15.html)
11. Vogels et al., “PowerSGD.” NeurIPS 2019.
    [Paper](https://papers.neurips.cc/paper_files/paper/2019/hash/d9fbed9da256e344c1fa46bb46c34c5f-Abstract.html)
12. Abrahamyan et al., “Learned Gradient Compression for Distributed Deep
    Learning.” 2021. [Paper](https://arxiv.org/abs/2103.08870)
13. Karimireddy et al., “Error Feedback Fixes SignSGD and other Gradient
    Compression Schemes.” ICML 2019.
    [Paper](https://proceedings.mlr.press/v97/karimireddy19a.html)
14. Rothchild et al., “FetchSGD.” ICML 2020.
    [Paper](https://proceedings.mlr.press/v119/rothchild20a.html)
