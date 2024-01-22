---
permalink: /docs/tacticzero/
title: "TacticZero Experiments"
---

TacticZero experiments use the HOL4 environment and dataset as outlined in the original paper (link to paper). 
It is an online policy gradient based algorithm. The architectures are defined in `models.TacticZero`, with the
agent interaction loop being defined in `experiments.TacticZero`.


# Running
To run a TacticZero experiment, from the root directory of the project simply run:

`python3 -m experiments.TacticZero.tacticzero_experiment --config-name=tacticzero/{model}`

# Configuration

Users can specify what architectures to use for the goal, tactic and argument selection modules through the
configurations in `configs/tacticzero`. Here, `configs/tacticzero/tactic_zero` is the base config, and the other config files
are used to define model details for the GNN, Transformer and original Autoencoder used.
Pretrained embedding models from Premise Selection experiments can be included by setting `pretrain: true` and `pretrain_ckpt`
to the path of the pretrained checkpoint.

