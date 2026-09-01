# OTS Compression Research

OTS stands for **Online Tensor Streaming**. This repository provides research
infrastructure for losslessly compressing gradient time series. A training run
writes an append-only, uncompressed trace; baseline and candidate compressors
consume the same bytes through a shared online/offline API.

## Repository areas

```text
src/ots_compression/
  core/                   # settings, experiment manifests, gradient trace format
  nn_training/
    runner.py             # shared optimizer/gradient-recording loop
    tasks/
      vision/             # CIFAR-10 and Cityscapes tasks/models
      nlp/                # WikiText causal language modeling
      speech/             # Speech Commands and compact Conformer
  compression/
    api.py                # online/offline compressor contract
    algorithms/           # baseline and future compressors
    evaluation.py         # correctness and performance experiments
  implementation_search/  # future agentic search

nn_training/configs/      # recipes plus the experiment meta-config/schema
baselines/                # baseline experiment notes/configuration
iter_impl_search_code/    # implementation-search notes/configuration
```

## Storage configuration

Copy `.env.example` to `.env` and set an absolute path, or export it directly:

```sh
export OTS_EXPERIMENTS_DIR=/Volumes/gradient-research/experiments
export OTS_DATASETS_DIR=/Volumes/gradient-research/datasets
```

Without an override, traces are written beneath `./experiments`.

## Development

```sh
python -m pip install -e '.[training,baselines,test]'
pytest
```

Run a training recipe and then all installed baselines:

```sh
ots-train --config nn_training/configs/vision/cifar10_small_convnet.json
ots-baselines --experiment experiment_001
ots-lossy --experiment experiment_001
```

Results are separated by fidelity regime: `lossless/<codec>/<mode>/` contains
exact-codec controls, while `lossy/<codec>/<mode>/` contains approximate
algorithms, decoded traces, and fidelity measurements.  See
[`iter_impl_search_code/`](iter_impl_search_code/) for the cumulative algorithm
index and iteration protocol.

When `training.online_compression` is configured, OTS-DeltaQ appends a complete
compressed record at every gradient-capture point. Its `online/` directory also
contains the decoder predictions and per-step prediction metrics used for
trajectory plots; these diagnostic predictions are not required by the decoder
and are not counted in the compression ratio.
