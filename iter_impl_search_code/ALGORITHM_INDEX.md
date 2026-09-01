# Implemented algorithm index

| ID | Status | Method | Parent | Code | Result/writeup |
| --- | --- | --- | --- | --- | --- |
| `L0` | complete | Exact raw, DEFLATE, Zstandard, LZ4, Snappy controls | — | `compression/algorithms/baselines.py` | `results/L0_lossless_controls.md` |
| `D1` | complete; replay passed | OTS-DeltaQ: decoder-consistent temporal residual blocks with int8/FP16/FP32 fidelity fallbacks | `L0` | `compression/algorithms/ots_deltaq.py` | `results/D1_ots_deltaq.md` |
| `D2` | complete; online replay passed | True training-time online D1 with saved decoder predictions and per-step prediction/reconstruction telemetry | `D1` | `compression/algorithms/ots_deltaq.py`, `nn_training/runner.py` | `results/D2_online_predictions.md` |
| `D3` | complete; rate hypothesis rejected | Zero-predictor DeltaQ ablation using the same adaptive int8/FP16/FP32 quantizer | `D2` | `compression/algorithms/ots_deltaq.py` | `results/D3_zero_predictor.md` |
| `D4` | complete; marginal rate gain, compute tradeoff rejected | Per-block adaptive selector between zero and previous-decoded prediction | `D3` | `compression/algorithms/ots_deltaq.py` | `results/D4_adaptive_predictor.md` |
| `C1` | planned control | Numcodecs `BitRound`/`Quantize`, tuned to the same energy-R2 gate | `D1` comparison control | — | — |

Read the listed writeups before extending an entry. New IDs are append-only;
superseded candidates remain documented with their negative findings.
