# G1: late-step causal structural rate frontier

## Hypothesis

G0 found that isolated low-rank, tensor, Fourier, and causal rank-8 models do
not plausibly reach the hard 100x target. G1 tests the narrow remaining
gradient-only hypothesis: a globally allocated mixture of asymmetric causal
subspaces, explicit basis innovations, quantized low-rank cores, and
sparse/bulk residuals can reconstruct a full step at `R2 >= 0.99` while the
**entire actual stream** remains at or below `0.32` bits per value.

This is a causal post-hoc feasibility screen. It is not an online training run.

## Method

For each of the 76 tensors in a selected step, the encoder enumerates a Pareto
frontier containing:

- zero prediction and dense affine residual quantization at 1, 2, or 4 bits;
- left-only, right-only, and two-sided projections onto the exact preceding
  gradient's SVD basis at ranks 4, 8, 16, 32, and 64;
- 4- or 8-bit quantized coefficients;
- aligned 4-bit coordinate innovations from the prior basis followed by QR,
  with an 8-bit core;
- 0.05%, 0.2%, and 1% sparse residual exceptions with delta-varint indices and
  8-bit values;
- selected sparse-outlier plus 4-bit bulk-residual combinations; and
- a stronger comparison frontier that may transmit paid 4- or 8-bit factors
  from the **current** SVD. These are labeled `oracle_svd` in the raw mode
  table because the post-hoc selector knows every candidate's exact error;
  their factors are not free.

All quantized payloads are bit-packed and kept raw or DEFLATE-coded according
to their actual length. Each tensor chunk contains deterministic JSON metadata.
A multiple-choice knapsack with a conservative 16-byte rate quantum selects
one self-contained chunk per tensor. The emitted stream is decoded, and all
reported fidelity numbers are recomputed from that decoded stream.

The decoder receives the tensor catalog and the exact previous gradient as
side information. To avoid treating the basis state as permanently free, each
causal chunk contains an uncompressed per-step charge equal to one hundredth
of an FP16 basis snapshot. The calculation nevertheless uses the exact FP32
prior basis, making this an optimistic upper bound. The global ledger includes
all chunk metadata, residual indices and values, factor/core payloads, and
these amortized charges.

The requested order was steps 100, 500, and 999. Global energy `R2 >= 0.99`
remains the scientific fidelity gate. Following the continuation decision, a
separate `R2 < 0.90` cutoff is the only condition that aborts the sampled run.

## Result

The input has 4,870,400 FP32 values and 19,481,600 uncompressed bytes per step.
The hard stream budget is 194,816 bytes.

| Step | Frontier | Bytes | bpv | Ratio | Energy R2 | Cosine | Encode | Decode |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | exact-prior causal only | 194,330 | 0.319202 | 100.250x | 0.988715 | 0.994343 | 6.986 s | 0.207 s |
| 100 | + paid current-SVD factors | 194,378 | 0.319281 | 100.225x | **0.993601** | 0.996796 | 7.189 s | 0.133 s |
| 500 | exact-prior causal only | 194,375 | 0.319276 | 100.227x | 0.960141 | 0.979883 | 6.792 s | 0.240 s |
| 500 | + paid current-SVD factors | 194,364 | 0.319258 | 100.233x | **0.970510** | 0.985163 | 7.009 s | 0.095 s |
| 999 | exact-prior causal only | 194,259 | 0.319085 | 100.287x | 0.934341 | 0.966629 | 7.160 s | 0.210 s |
| 999 | + paid current-SVD factors | 194,326 | 0.319195 | 100.252x | **0.948499** | 0.973936 | 6.971 s | 0.084 s |

The paid current-factor option makes step 100 feasible, and rank-64 plus
sparse/bulk candidates improve the causal-only result to within 0.001285 of
the fidelity gate. That success does not persist. At step 500 the strongest
tested frontier is 0.01949 below the R2 gate even though it uses 99.77% of the
byte budget and is allowed to mix current-SVD, exact-prior, innovation, sparse,
and dense modes tensor by tensor. The causal-only frontier is worse at
0.960141.

The structural controls from G0 remain far below this widened frontier: the
value-weighted rank-8 architecture-aware TT approximation captured 55.11% of
energy while its FP16 factors alone cost 0.2585 bpv, and Kronecker rank 8
captured 16.14%. They were not rerun as byte codecs because G1 already includes
the more favorable ordinary low-rank and low-rank-plus-sparse cases. At step
500, G0's free current rank-32 approximation captured 98.697% before factor
quantization or bytes; G1 shows that the globally allocated, paid, quantized
mixture reaches 97.051% under the complete ledger. By step 999, the widened
frontier declines further to 94.850%.

Step 500 failed the scientific fidelity gate but remained above the 0.90
continuation cutoff, so the run continued through step 999. Step 999 also
remained above the continuation cutoff. All three planned checkpoints were
therefore evaluated. Optimizer replay, endpoint weights, and final-model
distribution metrics were not run because the 0.99 protocol gate failed.
`final_model_outputs.csv` is intentionally absent because no replayed model
exists.

## Interpretation

G1 rejects the tested gradient-only structural mixture as the main route to a
uniform 100x archive. The result is stronger than G0's analytic screen because
it uses real decodable bytes and a global allocator, but it is not an
information-theoretic impossibility proof. Its favorable assumptions make the
negative late-step result especially informative:

1. The previous gradient and exact prior SVD bases are available to the
   decoder, while only a small amortized FP16 snapshot charge is paid.
2. Mode selection is post-hoc and sees the current gradient's exact errors.
3. The widened frontier can transmit current low-rank factors, not just reuse
   stale bases.
4. The result falls from 0.99360 at step 100 to 0.97051 at step 500 and
   0.94850 at step 999.

The limiting issue is late-training innovation, not merely factor overhead or
a poor fixed rank. At step 999, a more elaborate learned factor prior would
need to recover most of the missing 4.150 percentage points of full-gradient
energy at essentially zero extra rate. G0's weak temporal signal and G1's
exact-prior failure make that an unfavorable next bet.

The next experiment should be **G1P**, the separately labeled procedural
side-information control: regenerate gradients from batch/RNG/training state,
measure the transcript bytes and deterministic replay contract, and charge the
forward/backward decode time. Gradient-only G2/G3 work is not authorized by
this result.

## Reproduction

```bash
python -m iter_impl_search_code.experiments.G1.algorithm \
  --trace /Volumes/agocdrive/gradient-research/experiments/experiment_003/gradients.otsg \
  --steps 100,500,999 \
  --output iter_impl_search_code/experiments/G1/results.csv \
  --summary iter_impl_search_code/experiments/G1/summary.json
```

The trace is indexed in place. Only steps 99, 100, 499, 500, 998, and 999 are
decoded, and candidate generation completes for steps 100, 500, and 999. No
multi-gigabyte copy or compressed artifact is created.
