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
    initial_model.pt
    final_model.pt
    replay_contract.json
    lossless/<algorithm>/<online-or-offline>/metrics.json
    lossy/ots_deltaq_v1/online/
      compressed.otsdq
      predictions.otsg
      prediction_metrics.jsonl
      online_summary.json
```

`manifest.json` captures the domain, task, dataset, complete model/optimizer/
scheduler/training configuration, initialization, seed, environment, and a
stable configuration fingerprint.

`gradients.otsg` is a versioned append-only stream. Every step is a checksummed,
length-delimited record containing ordered tensor names, presence, dtype, shape,
and contiguous PyTorch bytes. It supports missing or changing tensors without
silently changing the format. It is deliberately uncompressed.

When `training.online_compression` is configured, the runner also encodes and
flushes each captured step before the optimizer advances. `predictions.otsg`
is optional diagnostic data containing the actual decoder-known prediction at
every step; it is not required to decode and is excluded from compression size.
The JSONL metrics provide the per-step prediction error trajectory directly.

By default, the shared runner records at `optimizer_input`: after AMP unscaling
and configured gradient clipping, immediately before the optimizer mutates or
clears gradients. These are the exact `.grad` tensors passed to
`optimizer.step()`. Set `training.gradient_capture` to `backward_output` to use
the earlier hook immediately after backward; with fp16 AMP, that earlier trace
still contains loss-scaled gradients.

## Configuration contract

[`configs/experiment.schema.json`](configs/experiment.schema.json) is the
meta-config for concrete recipes. Its referenced task, optimization, and run
schemas list every accepted categorical value, numeric bound, and
variant-specific sub-config. Checked-in recipes discover and validate against
it automatically; use `ots-train --meta-config PATH` for a recipe stored
elsewhere.

Optimizer branches are deliberately separate, so AdamW cannot silently accept
an SGD-only option such as `momentum`. Adam, AdamW, SGD, and RMSprop are runnable.
The catalog also sketches the planned nested MuonClip contract (Muon, fallback
AdamW, and QK-clip settings), but excludes it from runnable values until its
model-aware QK-clipping hooks are implemented.

## Implemented task matrix

Keep each run near 1–10M parameters and 1,000 optimizer steps:

| Domain | Benchmark | Model | Parameters |
|---|---|---:|---:|
| Vision | CIFAR-10 classification | small CNN | 1.15M |
| Vision | CIFAR-10 classification | tiny ViT | 2.69M |
| Vision | Cityscapes segmentation | small U-Net | 1.93M |
| NLP | WikiText-2 language modeling | causal Transformer | 4.87M |
| Audio | Speech Commands classification | small Conformer | 2.03M |
| NLP | WikiText-2 fine-tuning | TinyStories-1M | ~1M |

Tasks implement the shared `TrainingTask` interface; configuration and gradient
recording remain independent of domain.

The vision datasets are loaded through `torchvision`; Speech Commands is loaded
through `torchaudio`; WikiText is loaded through Hugging Face `datasets` because
modern TorchText distributions are not assumed. Cityscapes must be obtained
separately under `OTS_DATASETS_DIR` according to its dataset terms. The other
recipes request their dataset download on first use.

The TinyStories recipe is a real pretrained causal LM loaded through the
optional `transformers` training dependency. Set model architecture to
`huggingface_causal_lm`, provide `pretrained_model`, and optionally override
the matching `tokenizer` or pin `revision`.

Run any checked-in recipe with:

```sh
ots-train --config nn_training/configs/nlp/wikitext2_causal_transformer.json
```

The shared loop supports SGD/Adam/AdamW/RMSprop, constant/cosine/warmup-cosine
schedules, gradient accumulation, clipping, fp32/fp16/bf16 autocast, and CPU,
CUDA, or MPS.
