# Neural net training

This repo is used for training code in order to generate gradient histories for compression.

The idea for initial exploratory research is to train maybe a dozen ~10M parameter scale models for ~1000 steps each, to generate gradient histories.

We will store these gradient histories (~40GB per training run for `fp32` gradients).

Then we will run baseline lossless compression algorithms - both online compression over the step dimension and offline compression on the whole gradient array - and compare them on compression rate, speed, etc.

With that context in mind, in this neural net training repo specifically we will train a few types of models:

* 2D CNNs and ViT for vision tasks like image classification, semantic segmentation, etc. using benchmark datasets such as imagenet, cfar-10, cityscapes, coco, etc.

* Small transformers for natural language modeling 

* Small Conformer for audio tasks

* Maybe some others

we will train both from scratch and finetune pretrained models; we will use a variety of optimizers, learning rates, and other hyperparameter / training arg choices that could affect gradient distributions / dynamics, to test compression rates and speeds in various regimes, and to approximate what this might look like in a realistic pretraining/post-training run at the frontiers.

The outputs of these runs are append-only, uncompressed gradient histories
ready to be fed to compression algorithms. Built-in tasks use real benchmark
datasets only; there are no production synthetic or dummy-data paths.

We could in principle save the whole optimizer state, which could be helpful for when optimizer-specific info like momentum is valuable, but that is more of a production-grade concern; for these research purposes it is enough to just save ragged gradient arrays.

## Experiment layout

The root is selected by `OTS_EXPERIMENTS_DIR` in the environment or `.env`:

```text
<experiments-root>/
  experiment_001/
    manifest.json
    gradients.otsg
    baselines/<algorithm>/<online-or-offline>/metrics.json
```

`manifest.json` captures the domain, task, dataset, complete model/optimizer/
scheduler/training configuration, initialization, seed, environment, and a
stable configuration fingerprint.

`gradients.otsg` is a versioned append-only stream. Every step is a checksummed,
length-delimited record containing ordered tensor names, presence, dtype, shape,
and contiguous PyTorch bytes. It supports missing or changing tensors without
silently changing the format. It is deliberately uncompressed.

The shared runner records immediately after backward/unscaling/clipping and
before the optimizer mutates or clears gradients.

## Implemented task matrix

Keep each run near 1–10M parameters and 1,000 optimizer steps:

| Domain | Benchmark | Model | Parameters |
|---|---|---:|---:|
| Vision | CIFAR-10 classification | small CNN | 1.15M |
| Vision | CIFAR-10 classification | tiny ViT | 2.69M |
| Vision | Cityscapes segmentation | small U-Net | 1.93M |
| NLP | WikiText-2 language modeling | causal Transformer | 4.87M |
| Audio | Speech Commands classification | small Conformer | 2.03M |

Tasks implement the shared `TrainingTask` interface; configuration and gradient
recording remain independent of domain.

The vision datasets are loaded through `torchvision`; Speech Commands is loaded
through `torchaudio`; WikiText is loaded through Hugging Face `datasets` because
modern TorchText distributions are not assumed. Cityscapes must be obtained
separately under `OTS_DATASETS_DIR` according to its dataset terms. The other
recipes request their dataset download on first use.

Run any checked-in recipe with:

```sh
ots-train --config nn_training/configs/nlp/wikitext2_causal_transformer.json
```

The shared loop supports SGD/Adam/AdamW/RMSprop, constant/cosine/warmup-cosine
schedules, gradient accumulation, clipping, fp32/fp16/bf16 autocast, and CPU,
CUDA, or MPS. For fp16, gradients are unscaled before recording. If clipping is
configured, the saved values are the clipped gradients actually consumed by the
optimizer.
