---
permalink: /docs/int/
title: "INT Experiments"
---

The INT experiments are adapted from the code published in the paper. 
The primary change is the additional compatibility of Transformer Encoder models with the
graph based observation mode.
Previously, a different environment observation was used with Transformer based models, 
and it was only compatible with Encoder-Decoder Transformers.
These additions enable a better comparison of embedding architectures, where the rest of the setup is held constant.
Experiments using the other sequence based representation could be integrated with some additional effort.

The architectures are defined in `models.INT`.


# Running
To run an INT experiment, from the root directory of the project run:

`python3 -m experiments.INT.int_experiment --config-name=int/int_base`

# Configuration

The configuration file for this experiment is given in `configs/INT/int_base.yaml`.

Most of the parameters follow the documentation from the original repo. 
The primary change is the `obs_mode` key which controls the embedding model used.

It can be either `geometric` for GNN models, or `seq` to use a Transformer Encoder.


# Data
- INT Data 

This experiment assumes data has been processed following the instructions from the original repo,
using the code in `environments.INT`

# Environments
- INT

# Models
- INT models with GNN and Transformer Encoders, as defined in `models.INT`.
