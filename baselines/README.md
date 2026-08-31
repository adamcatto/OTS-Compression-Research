# Baseline compression algorithms

Baseline causal-online and full-trace compressors.

## Access regimes

- **Online:** the encoder receives one complete gradient step at a time, may
  retain causal state, and must checkpoint output before seeing the next step.
- **Offline:** preparation may inspect/replay the completed trace. Encoding is
  still streamed so a 40 GB trace need not be resident in memory.

The regime is independent of the codec. Raw, DEFLATE, Zstandard, LZ4, and
Snappy use the same `Compressor`/`Encoder`/`Decoder` API. A full-trajectory
method is only called an oracle when it actually uses future information.

```python
from ots_compression.compression import AccessMode
from ots_compression.compression.algorithms import DeflateCompressor
from ots_compression.compression.evaluation import benchmark_experiment

metrics = benchmark_experiment(
    experiment,
    DeflateCompressor(level=6),
    AccessMode.ONLINE,
)
```

Every run performs an exact byte-for-byte decode check. Metrics include total
and payload size, side information, compression ratio, space saving, preparation/
encode/decode time, throughput, online step latency, average/peak process RSS,
and an optional operation/FLOP estimate. FLOPs remain `null` for library codecs
whose work is dominated by integer and byte operations and lacks trustworthy
instrumentation.

Run every installed baseline in both regimes:

```sh
ots-baselines --experiment experiment_001
```
