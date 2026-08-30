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

The outputs of these runs will be files with uncompressed gradient histories, stored as ragged PyTorch tensors ready to be fed to compression algorithms.

We could in principle save the whole optimizer state, which could be helpful for when optimizer-specific info like momentum is valuable, but that is more of a production-grade concern; for these research purposes it is enough to just save ragged gradient arrays.
