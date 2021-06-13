# CommutativeLieGroupVAE-Pytorch

Code for our paper [Commutative Lie Group VAE for Disentanglement Learning](https://arxiv.org/abs/2106.03375).

## Abstract

We view disentanglement learning as discovering an underlying structure that
equivariantly reflects the factorized variations shown in data.
Traditionally, such a structure is fixed to be a vector space with data
variations represented by translations along individual latent dimensions.
We argue this simple structure is suboptimal since it requires the model
to learn to discard the properties (e.g. different scales of changes,
different levels of abstractness) of data variations, which is an extra
work than equivariance learning. Instead, we propose to encode the data
variations with groups, a structure not only can equivariantly represent
variations, but can also be adaptively optimized to preserve the properties
of data variations. Considering it is hard to conduct training on group
structures, we focus on Lie groups and adopt a parameterization using
Lie algebra. Based on the parameterization, some disentanglement learning
constraints are naturally derived. A simple model named Commutative Lie Group
VAE is introduced to realize the group-based disentanglement learning.
Experiments show that our model can effectively learn disentangled
representations without supervision, and can achieve state-of-the-art
performance without extra constraints.

## Requirements

* Python == 3.6.12
* Numpy == 1.19.2
* PyTorch == 1.7.1
* Our code is based on
[this](https://github.com/MattPainter01/UnsupervisedActionEstimation)
repository, thus has the same following structure:
- datasets: Stores PyTorch datasets and code to initialise them.
- logger: Stores tensorboard logging and image generation code.
- metrics: Stores all disentanglement metrics.
- models: Stores all models used.
- main.py: Defines the command line args to run training and executes the trainer.
- trainer.py: Sets up parameters for training, optimisers etc.
- training_loop.py: Defines the model independent training logic.
