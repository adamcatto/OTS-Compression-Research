# F lineage: streaming structure, rate-distortion, and dynamics

This note records the literature synthesis used to start a new algorithm
lineage after E2. It is deliberately scoped to gradient archives that must be
decoded without training data or activations. Communication compressors are
evidence, not drop-in solutions: they usually optimize convergence of a live
training job rather than fidelity of every reconstructed gradient and model
replay.

## Constraints that change the problem

- A step is encoded when it is captured. A bounded, explicitly reported delay
  may be explored later, but no future gradient is available to the default
  encoder.
- The decoder starts with the initial model, optimizer contract, and artifact.
  It does not have minibatches, activations, backpropagated errors, or a hidden
  pretrained compression model.
- Global state proportional to parameter count is not viable at trillion-model
  scale. State must be shard-local and sublinear, ideally
  `O(r * sum(matrix dimensions) + sketch width)`.
- Gradient energy-R2 must exceed 0.99, but that is only an early gate. AdamW
  replay, cumulative bias, final weights, loss, and output KL/JS decide whether
  an algorithm is useful.
- A trillion FP32 gradient contains 4 TB. Even ideal coding of independent
  Gaussian coordinates at normalized MSE 0.01 has a rate-distortion lower
  bound of about 3.32 bits/value. Exact minibatch noise therefore creates a
  real floor; structure can beat it only to the extent that the stream is not
  memoryless isotropic noise.

### Formal objective

Let `g_t` be the exact optimizer-input gradient at step `t`, `g_hat_t` the
decoded gradient, and `C_t` the emitted code. The immediate distortion gate is

```text
R2_t = 1 - ||g_t - g_hat_t||_2^2 / ||g_t||_2^2 >= 0.99.
```

The research objective is not simply `min sum_t |C_t|` under this constraint.
AdamW is a stateful nonlinear map. Ignoring bias correction and weight decay
for notation, its moments and update are

```text
m_t = beta1 m_(t-1) + (1-beta1) g_t
v_t = beta2 v_(t-1) + (1-beta2) g_t^2
u_t = m_t / (sqrt(v_t) + epsilon).
```

A locally small, biased error can perturb both numerator and denominator for
every later update. To first order, an error `e_t = g_hat_t - g_t` contributes
directly through a diagonal sensitivity related to `1 / sqrt(v_t)` and
indirectly through the two recurrent moment states. D5 is an empirical example:
gradient R2 was 0.999952 and replay loss MAE was only `5.65e-5`, yet the final
weight relative L2 error reached 2.86%. The operational objective is therefore

```text
minimize  sum_t |C_t|
subject to
  R2_t >= R2_min,
  cumulative optimizer-update error <= U_max,
  low-dimensional signed error debt <= B_max,
  state and work fitting the shard-local scaling bounds.
```

Output KL/JS and final weights act as end-to-end constraints that the local
surrogates may fail to predict.

### Information-theoretic floor

For an iid Gaussian source with variance `sigma^2` and squared-error distortion
`D`, Shannon's scalar rate-distortion function is
`R(D) = 0.5 log2(sigma^2 / D)` bits/value. An R2 floor of 0.99 corresponds to
`D / sigma^2 <= 0.01`, hence at least 3.32 bits/value even before headers,
finite-block penalties, random access, or nonstationarity. A one-trillion-value
step would still require roughly 415 GB at that idealized lower bound. This
does not prove real gradients need that rate—the entire project is testing
their structure—but it prevents an implausible assumption that trillion-scale
archives can become tiny while retaining 99% of arbitrary minibatch-gradient
energy. [TurboQuant](https://arxiv.org/abs/2504.19874) is useful here because it
compares a practical online vector quantizer directly with information-theoretic
distortion bounds rather than reporting only downstream accuracy.

## Literature threads and what transfers

### Streaming sketches and subspace tracking

[Frequent Directions](https://research.ibm.com/publications/frequent-directions-simple-and-deterministic-matrix-sketching)
maintains a mergeable deterministic low-rank matrix sketch with bounded space.
[Oja's algorithm](https://proceedings.mlr.press/v49/jain16.html) provides a
single-pass `O(dk)`-space route to streaming PCA. GROUSE tracks a subspace with
Grassmannian rank-one updates, including from incomplete observations
([paper](https://people.eecs.berkeley.edu/~brecht/papers/10.Bal.Now.Rec.GROUSE.pdf)),
while PETRELS uses discounted recursive least squares to follow nonstationary
subspaces
([paper](https://users.ece.cmu.edu/~yuejiec/papers/PETRELS_TSP.pdf)).
[FetchSGD](https://proceedings.mlr.press/v119/rothchild20a.html) is especially
relevant because Count Sketch is linear enough to hold momentum and error
accumulation in sketch space instead of ambient parameter space.

Transfer: maintain bases or error memory from the decoded stream so encoder and
decoder states cannot drift. Use mergeable sketches for innovations and global
telemetry. Do not maintain a dense residual per parameter.

### Gradient compression and accumulated error

[PowerSGD](https://papers.neurips.cc/paper_files/paper/2019/hash/d9fbed9da256e344c1fa46bb46c34c5f-Abstract.html)
shows that matrix gradients admit fast low-rank compression and that warm-started
power iteration is practical. [Error-feedback SGD](https://proceedings.mlr.press/v97/karimireddy19a.html)
shows why biased local quantization errors cannot simply be discarded.
[Deep Gradient Compression](https://arxiv.org/abs/1712.01887) combines sparse
updates with momentum correction, clipping, masking, and warmup. [1-bit
Adam](https://proceedings.mlr.press/v139/tang21a.html) shows that Adam's
variance state can stabilize after warmup, after which a fixed preconditioner
makes aggressive error-compensated compression practical.

Transfer: E1/E2 already validate the PowerSGD premise locally. D5 validates the
error-feedback warning: excellent per-gradient R2 can coexist with accumulated
endpoint drift. Future allocation should measure optimizer-preconditioned error
and low-dimensional error debt, not just instantaneous Frobenius error.

### LLM gradient dynamics

[GaLore](https://arxiv.org/abs/2403.03507) projects full-parameter LLM gradients
into low-rank spaces. [Q-GaLore](https://arxiv.org/abs/2407.08296) reports that
subspace stability is heterogeneous across layers, many subspaces can be
updated lazily, projection matrices tolerate low-bit storage, and stochastic
rounding is important for preserving small accumulated updates.
[SubTrack-Grad](https://arxiv.org/abs/2502.01586) replaces repeated SVDs with
Grassmannian tracking. [APOLLO](https://proceedings.mlsys.org/paper_files/paper/2025/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf)
finds that low-rank random projections suffice to estimate much coarser
channel- or tensor-level Adam scaling. Earlier work found that gradients can
concentrate in a small, persistent top-Hessian subspace
([Gur-Ari et al.](https://arxiv.org/abs/1812.04754)).

Transfer: use layer-adaptive refresh based on measured subspace innovation;
avoid periodic SVD and model-sized learned state. Coarse optimizer sensitivity
can guide bit allocation without recreating full Adam statistics inside the
codec.

### Quantization and rate-distortion

[EDEN](https://proceedings.mlr.press/v162/vargaftik22a.html) and
[QUIC-FL](https://arxiv.org/abs/2205.13341) use randomized Hadamard transforms
to spread coordinate energy before scalar quantization; the transform is
orthogonal, parallel, in-place, and `O(d log d)`. [TurboQuant](https://arxiv.org/abs/2504.19874)
connects random rotation and optimized scalar codebooks to near-optimal online
vector-quantization distortion. Classical error-feedback quantization also has
an explicit rate-distortion interpretation
([analysis](https://arxiv.org/abs/1609.01383)).

Transfer: transform heavy-tailed residual blocks before quantization so a
single maximum no longer dictates every coordinate's scale. Preserve the
orthogonal transform seed in the format rather than storing rotations.

### Learned compression, physics, and mechanistic interpretation

[Learned Gradient Compression](https://arxiv.org/abs/2103.08870) demonstrates
that lightweight autoencoders can exploit inter-worker correlations. Online
continual compression exposes the harder versioning problem: old codes must
remain decodable after the model changes
([Caccia et al.](https://proceedings.mlr.press/v119/caccia20a.html)). A learned
autoencoder per tensor family is therefore not the first scalable choice here.

K-FAC's successful Kronecker factorization of layerwise Fisher blocks
([Martens and Grosse](https://proceedings.mlr.press/v37/martens15.html)) supports
two-sided matrix statistics. The neural tangent kernel describes a regime in
which function dynamics are much lower-dimensional than parameter space
([Jacot et al.](https://arxiv.org/abs/1806.07572)); random-matrix optimal
singular-value shrinkage separates spectral spikes from a noise bulk
([Gavish and Donoho](https://arxiv.org/abs/1405.7511)). Mechanistic
interpretability's superposition model says meaningful features are directions,
not reliably aligned individual coordinates
([Elhage et al.](https://www.anthropic.com/research/toy-models-of-superposition)).
Transformer and feed-forward parameterizations also have gauge/permutation
redundancies ([Hashimoto et al.](https://arxiv.org/abs/2402.02362)).

Transfer: spectral spikes and tensor-local two-sided bases are more defensible
than copying raw coordinates between layers. Cross-layer sharing should begin
with scalar control statistics or sketches, not a shared raw-coordinate basis,
unless an explicit alignment is learned and transmitted.

## Reasoning from our previous iterations

The new line is constrained by five measurements from D2–E2:

1. D2's previous-decoded-gradient predictor has mean whole-step prediction R2
   `-0.4719`; raw minibatch-to-minibatch copying is not a useful global temporal
   model. It nevertheless enables int8 for more large-block elements than the
   zero predictor, so local rate is not determined by whole-step predictor R2.
2. D4's per-block predictor selection saves only 0.19% of bytes while adding
   81% encode time. More candidate evaluation is worthwhile only if it changes
   the representation, not merely the temporal baseline.
3. D5's sparse outliers raise compression from about 2.02x to 3.09x, proving
   dynamic-range concentration is the main quantizer failure. Its endpoint
   drift shows that exception selection/rounding also introduces harmful
   cumulative structure.
4. E1/E2 show real matrix structure: rank-1 predictions move 0.89B/1.13B
   values to int8 and warm tracking improves selection by 29.2%. Factor bytes
   are negligible, but matrix structure alone leaves most values in FP16.
5. The final-model evaluation sharpens the endpoint story. E2 retains 99.852%
   top-1 agreement and reference-to-lossy KL `7.96e-6`; D5 retains 99.720% but
   KL rises to `5.06e-5` and logit relative L2 to 11.8%. Scalar training loss
   can conceal a much larger output-space shift.

These facts favor a composition in which rotations handle the noise-like
residual, subspaces handle spectral spikes, and a signed temporal controller
prevents systematic error accumulation. They argue against betting the next
iteration on a deeper autoregressive predictor or a larger rank alone.

## Why F1 uses randomized Hadamard mixing

For a power-of-two block `x` of length `n`, define

```text
y = R x,  R = H D,
```

where `D` is a deterministic seeded Rademacher diagonal and `H` is the
orthonormal Walsh-Hadamard matrix. Since `R^T R = I`, quantization error measured
after inverse rotation is exactly the transformed-coordinate error:

```text
||x - R^T Q(Rx)||_2 = ||Rx - Q(Rx)||_2.
```

The transform cannot manufacture a better L2 error for an optimal vector
quantizer. Its value is that F1 uses a max-scaled scalar int8 quantizer. Without
mixing, one extreme coordinate sets the quantization step for the entire block.
With random signs, each transformed coordinate is a signed sum of all input
coordinates and is sub-Gaussian with variance on the order of
`||x||_2^2 / n`. Consequently the maximum is typically
`O(||x||_2 sqrt(log n / n))`, rather than `||x||_infinity`. A uniform int8
quantizer then has the rough normalized error scaling
`O(log n / 127^2)`, instead of an error controlled by the original peak-to-RMS
ratio. This is the same concentration mechanism exploited by randomized
Hadamard gradient/mean estimators in
[EDEN](https://proceedings.mlr.press/v162/vargaftik22a.html) and
[QUIC-FL](https://arxiv.org/abs/2205.13341).

F1 deliberately uses the zero predictor. That makes the transformed residual
equal to the current gradient and isolates the transform from D2's sometimes
helpful but globally poor temporal residual. It also removes the risk that a
large prediction error consumes the strict block budget. F2 can restore
multiple candidates under a global allocator after F1 tells us whether the
transform is worth its compute cost.

## Proposed scalable stack

The target design is **SPECTRA**: Streaming Predictive Error-Controlled Tensor
Residual Archive.

1. **Orthogonal residual mixing.** Apply a seeded randomized Hadamard transform
   independently to fixed-size residual blocks, then use a symmetric or
   Gaussian-optimized scalar quantizer. This is stateless and preserves L2
   distortion exactly before quantization.
2. **Decoder-synchronized tensor subspaces.** For each large matrix shard,
   retain small left/right bases and update them from the decoded gradient using
   Oja/GROUSE-style steps. Encode only a small core and innovation. State is
   `O(r(m+n))`, not `O(mn)`, and the decoder never needs current activations.
3. **Sketch-space error debt.** Accumulate Count-Sketch projections of
   reconstruction error and optimizer-preconditioned error. Promote persistent
   heavy directions or bias corrections in later records. This creates
   non-greedy behavior without a full-size residual buffer.
4. **Global rate-distortion allocation.** Each block exposes a small candidate
   curve (bits, gradient SSE, projected update error). A streaming Lagrange
   multiplier chooses modes across the whole shard/step under the global R2
   budget and a cumulative error-debt budget. This replaces today's much more
   conservative independent per-block gate.
5. **Slow learned controller, not learned reconstruction.** A tiny online model
   may predict rank, refresh cadence, transform depth, and quantizer from
   telemetry. The payload remains decodable from explicit deterministic codec
   state; changing the controller never invalidates old records.

### Non-greedy rate allocation

For block `i`, suppose the encoder can construct candidates `c` with byte cost
`b_ic`, gradient SSE `d_ic`, and optimizer-weighted proxy error `a_ic`. Instead
of applying the same distortion threshold independently to every block, F2
will solve the separable Lagrangian

```text
choose c_i minimizing sum_i [b_i,c_i + lambda d_i,c_i + mu a_i,c_i]
```

while adapting `lambda` and `mu` until the shard/global distortion and error
debt constraints are met. Each block still makes a constant-time choice from a
small frontier, so the implementation is linear and parallel. The multiplier
couples choices across tensors: a noisy, insensitive block may spend more error
so a coherent or optimizer-sensitive block can spend less. A carried bit/error
reservoir extends that coupling across steps without buffering future
gradients. This is non-greedy in rate-distortion space even though encoding
remains causal.

### Decoder-synchronized subspace equations

For a matrix gradient shard `G_t in R^(m x n)`, maintain bases
`U_(t-1) in R^(m x r)` and `V_(t-1) in R^(n x r)` derived only from previously
decoded gradients. Encode the small core

```text
C_t = U_(t-1)^T G_t V_(t-1)
G_signal,t = U_(t-1) C_t V_(t-1)^T,
E_t = G_t - G_signal,t.
```

The core costs `O(r^2)` values; the residual goes through rotated
rate-distortion coding. After decoding `G_hat_t`, both sides apply the same
Oja/GROUSE update and re-orthogonalization to obtain `U_t,V_t`. New directions
enter through the high-fidelity residual, so the tracker is not locked to its
initial span. An innovation statistic such as
`||E_t||_F^2 / ||G_t||_F^2` controls rank or refresh cadence, following
Q-GaLore's empirical finding that different layers stabilize at different
times ([paper](https://arxiv.org/abs/2407.08296)).

Raw bases are tensor-local. Sharing them across transformer layers would assume
coordinate alignment that our E1 study did not find and that neural-network
gauge/permutation symmetries make theoretically suspect. Cross-layer sharing is
limited initially to dimensionless scalars—spectral entropy, innovation ratio,
quantizer histograms, and controller parameters.

### Compact temporal error state

Full error feedback stores `e_t` for every parameter, an unacceptable extra
model copy at trillion scale. A Count Sketch `S e_t` uses a fixed number of
counters and is linear:

```text
S(e_t + e_(t-1)) = S e_t + S e_(t-1).
```

This permits signed error accumulation, heavy-coordinate recovery, and
mergeable shard summaries without dense state, as demonstrated for momentum
and error accumulation by
[FetchSGD](https://proceedings.mlr.press/v119/rothchild20a.html). The limitation
is equally important: a sketch cannot reconstruct a dense isotropic error.
SPECTRA therefore uses it to detect persistent bias/heavy directions and to
steer later bits or subspace innovations, not to pretend that all omitted noise
can be recovered.

For an `m x n` shard, transform work is `O(mn log B)` for block size `B`, basis
work is `O(rmn)`, persistent basis state is `O(r(m+n))`, and sketch state is
fixed by the chosen error budget. No collective operation or whole-model tensor
assembly is required. At trillion scale the algorithm runs independently on
the same parameter shards already established by ZeRO/tensor parallelism;
[ZeRO](https://doi.org/10.1109/sc41405.2020.00024) and
[Megatron-LM](https://arxiv.org/abs/2104.04473) demonstrate why respecting those
shards is a system requirement rather than an implementation detail.

### Scaling example and bottlenecks

For square width `d` and rank `r`, basis state per matrix is `2dr` values versus
`d^2` parameters, a fraction `2r/d`. At `d=16,384` and `r=8`, FP16 bases are
about 0.098% of the matrix's parameter count. This favorable asymptotic does not
make the method free: basis projection costs `O(r d^2)` arithmetic and the
Hadamard transform makes `log2(B)` passes over every transformed block. A
production trillion-scale design therefore needs fused accelerator kernels,
overlap with gradient sharding/I/O, and adaptive skipping when a transform or
subspace is not paying for itself. The Python timings in this repository are
codec comparisons, not claims of production throughput.

The archive format must also checkpoint deterministic state or make it
reconstructible by scanning from a prior checkpoint. Random access every `K`
steps requires periodic basis/sketch snapshots with overhead amortized across
the interval. This is another reason to avoid an opaque learned decoder whose
weights evolve continuously.

## Alternatives considered and rejected for the first F iteration

- **A gradient autoencoder.** Learned Gradient Compression supports the
  existence of exploitable correlations, but training and versioning a decoder
  per architecture introduces opaque state, expensive inference, and old-code
  compatibility before simpler structure has been exhausted.
- **Higher-rank PowerSGD only.** E1/E2 already establish this direction.
  Increasing rank may improve rate, but it does not address the dominant
  heavy-tailed residual or endpoint bias and would not be a genuinely new
  mechanism.
- **Full dense error feedback.** It is effective but adds one parameter-sized
  state per encoder shard; globally that is another trillion values.
- **Raw cross-layer prediction.** E1 measured only 0.059 energy-weighted fitted
  R2 for homologous preceding-layer tensors, and layer coordinate systems have
  symmetries. Sharing scalar statistics is a safer first test.
- **A delayed multi-step SVD.** It could be noncausally optimal over a window,
  but it delays archival durability and needs multiple gradient-sized buffers.
  The causal multiplier/error-debt formulation obtains non-greedy behavior with
  bounded sublinear state instead.

## Experimental sequence

- **F1 (complete):** randomized-Hadamard residual blocks plus the existing
  strict DeltaQ fallback. It reached 3.447x and transformed 4.101B values into
  int8, validating outlier flattening, but final-weight relative L2 reached
  6.96% and output KL reached `8.78e-4`. The standalone endpoint hypothesis is
  rejected.
- **F2 (complete):** optimizer-aware global rate-distortion allocation. It
  reached 3.993x and 0.999916 gradient R2, but the weighted constraint remained
  slack (mean proxy relative SSE `8.82e-5` under a `1e-4` limit), leaving only
  82,176 FP16 values in 4.8704B. Final-weight relative L2 was 7.01% and output
  KL was `9.08e-4`, both slightly worse than F1. Rate won; endpoint fidelity
  and the chosen sensitivity-budget calibration were rejected.
- **F2R (complete; partial success):** rate-matched sensitivity allocation. A
  carried byte reservoir matched F1's 5,644,610,904-byte block payload exactly.
  At the same rate it improved gradient R2 from 0.999930143 to 0.999935095,
  reduced cumulative error L2 by 3.52%, final-weight relative L2 by 1.68%, and
  output KL by 1.81%. Top-1 agreement was essentially flat/slightly worse and
  the final-weight error remained 6.84%, so retain the placement signal without
  treating it as an endpoint solution.
- **F3:** decoder-synchronized rank-k tensor subspaces with no per-step full
  factors; compare fixed cadence with innovation-triggered refresh and feed its
  core/residual candidates into F2's allocator.
- **F4:** sketch-space signed error debt and persistent-direction correction,
  judged primarily by final weights and output KL/JS rather than another small
  per-step rate gain.

The sequence changed after F1. Its scalar error mean is essentially zero and
its raw cumulative error L2 is only 15.6% above D5, yet its final-weight error
is 2.43x D5's. A global scalar bias correction is therefore unlikely to be
enough. The discrepancy points to coordinate/tensor-dependent Adam sensitivity,
which is why F2 moves optimizer-aware distortion ahead of a looser global R2
budget or a larger structural model.

The sequence changed again after F2. Its global allocator exposed how
conservative F1's independent block gates were, but the identically valued raw
and weighted budgets were not comparably tight: transformed int8 usually met
both, so the optimizer proxy selected almost no extra precision. F2R must
rate-match F1 before F3; otherwise a comparison would confound sensitivity
placement with F2's 13.7% byte reduction and 20.3% larger raw error. If F2R
fails, the block-scalar proxy has earned rejection and the subspace/error-debt
stages become the next evidence-based move.

F2R completed that falsification. At exactly F1's block payload, optimizer-aware
placement modestly improved continuous replay and output-distribution metrics,
so the signal is retained. The limited effect and persistent concentration in
embedding/early-attention tensors rule out more scalar-budget tuning as the
main next step. F3 should combine decoder-synchronized tensor subspaces with
F2R's rate controller; F4 remains the signed-error-debt stage if structured
representation alone does not control trajectory accumulation.

## Selected primary references

1. Vogels, T., Karimireddy, S. P., and Jaggi, M. “PowerSGD: Practical
   Low-Rank Gradient Compression for Distributed Optimization.” NeurIPS 2019.
   [Paper](https://papers.neurips.cc/paper_files/paper/2019/hash/d9fbed9da256e344c1fa46bb46c34c5f-Abstract.html)
2. Karimireddy, S. P. et al. “Error Feedback Fixes SignSGD and other Gradient
   Compression Schemes.” ICML 2019.
   [Paper](https://proceedings.mlr.press/v97/karimireddy19a.html)
3. Rothchild, D. et al. “FetchSGD: Communication-Efficient Federated Learning
   with Sketching.” ICML 2020.
   [Paper](https://proceedings.mlr.press/v119/rothchild20a.html)
4. Ghashami, M., Liberty, E., Phillips, J. M., and Woodruff, D. P. “Frequent
   Directions: Simple and Deterministic Matrix Sketching.” SIAM Journal on
   Computing, 2016.
   [Paper](https://research.ibm.com/publications/frequent-directions-simple-and-deterministic-matrix-sketching)
5. Jain, P. et al. “Streaming PCA: Matching Matrix Bernstein and Near-Optimal
   Finite Sample Guarantees for Oja's Algorithm.” COLT 2016.
   [Paper](https://proceedings.mlr.press/v49/jain16.html)
6. Balzano, L., Nowak, R., and Recht, B. “Online Identification and Tracking
   of Subspaces from Highly Incomplete Information.” Allerton 2010.
   [Paper](https://people.eecs.berkeley.edu/~brecht/papers/10.Bal.Now.Rec.GROUSE.pdf)
7. Zhao, J. et al. “GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
   Projection.” 2024. [Paper](https://arxiv.org/abs/2403.03507)
8. Zhang, Z. et al. “Q-GaLore: Quantized GaLore with INT4 Projection and
   Layer-Adaptive Low-Rank Gradients.” 2024.
   [Paper](https://arxiv.org/abs/2407.08296)
9. Rajabi, S., Nonta, N., and Rambhatla, S. “SubTrack your Grad: Gradient
   Subspace Tracking for Memory and Time Efficient Full-Parameter LLM
   Training.” 2025. [Paper](https://arxiv.org/abs/2502.01586)
10. Zhu, H. et al. “APOLLO: SGD-like Memory, AdamW-level Performance.” MLSys
    2025. [Paper](https://proceedings.mlsys.org/paper_files/paper/2025/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf)
11. Vargaftik, S. et al. “EDEN: Communication-Efficient and Robust Distributed
    Mean Estimation for Federated Learning.” ICML 2022.
    [Paper](https://proceedings.mlr.press/v162/vargaftik22a.html)
12. Ben Basat, R. et al. “QUIC-FL: Quick Unbiased Compression for Federated
    Learning.” 2022. [Paper](https://arxiv.org/abs/2205.13341)
13. Zandieh, A. et al. “TurboQuant: Online Vector Quantization with Near-optimal
    Distortion Rate.” 2025. [Paper](https://arxiv.org/abs/2504.19874)
14. Martens, J. and Grosse, R. “Optimizing Neural Networks with
    Kronecker-Factored Approximate Curvature.” ICML 2015.
    [Paper](https://proceedings.mlr.press/v37/martens15.html)
15. Jacot, A., Gabriel, F., and Hongler, C. “Neural Tangent Kernel: Convergence
    and Generalization in Neural Networks.” NeurIPS 2018.
    [Paper](https://arxiv.org/abs/1806.07572)
16. Gur-Ari, G., Roberts, D. A., and Dyer, E. “Gradient Descent Happens in a
    Tiny Subspace.” 2018. [Paper](https://arxiv.org/abs/1812.04754)
17. Gavish, M. and Donoho, D. L. “Optimal Shrinkage of Singular Values.” IEEE
    Transactions on Information Theory, 2017.
    [Paper](https://arxiv.org/abs/1405.7511)
18. Elhage, N. et al. “Toy Models of Superposition.” Transformer Circuits
    Thread, 2022.
    [Paper](https://www.anthropic.com/research/toy-models-of-superposition)
