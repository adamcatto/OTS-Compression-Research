# Implemented algorithm index

| ID | Status | Method | Parent | Code | Result/writeup |
| --- | --- | --- | --- | --- | --- |
| `L0` | complete | Exact raw, DEFLATE, Zstandard, LZ4, Snappy controls | — | `experiments/L0/algorithm.py` | `experiments/L0/README.md` |
| `D1` | complete; replay passed | OTS-DeltaQ: decoder-consistent temporal residual blocks with int8/FP16/FP32 fidelity fallbacks | `L0` | `experiments/D1/algorithm.py` | `experiments/D1/README.md` |
| `D2` | complete; online replay passed | True training-time online D1 with saved decoder predictions and per-step prediction/reconstruction telemetry | `D1` | `experiments/D2/algorithm.py` | `experiments/D2/README.md` |
| `D3` | complete; rate hypothesis rejected | Zero-predictor DeltaQ ablation using the same adaptive int8/FP16/FP32 quantizer | `D2` | `experiments/D3/algorithm.py` | `experiments/D3/README.md` |
| `D4` | complete; marginal rate gain, compute tradeoff rejected | Per-block adaptive selector between zero and previous-decoded prediction | `D3` | `experiments/D4/algorithm.py` | `experiments/D4/README.md` |
| `D5` | complete; rate win, strict weight agreement rejected | Robust int8 residuals with sparse FP16 outlier exceptions | `D4` | `experiments/D5/algorithm.py` | `experiments/D5/README.md` |
| `E1` | complete; modest rate win, intermediate endpoint drift | Per-matrix learned rank-1 predictor with FP16 factors and strict DeltaQ residual fallback | structural branch from `D2`/`D5` findings | `experiments/E1/algorithm.py` | `experiments/E1/README.md` |
| `E2` | complete; rate win, endpoint drift slightly worse | Temporally warm-started rank-1 subspace tracking with E1's strict residual fallback | `E1` | `experiments/E2/algorithm.py` | `experiments/E2/README.md` |
| `C1` | planned control | Numcodecs `BitRound`/`Quantize`, tuned to the same energy-R2 gate | `D1` comparison control | — | — |

Read the listed writeups before extending an entry. New IDs are append-only;
superseded candidates remain documented with their negative findings.
