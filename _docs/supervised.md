---
permalink: /docs/supervised/
title: "Supervised Experiments"
---

Supervised Experiments are handled by `experiments.supervised.supervised_experiment_runner`.

Such Experiments include Premise Selection, the HOList training approach, 
Supervised pre-training and finetuning (as seen in ReProver), as well as some currently 
experimental experiments such as Direct Preference Optimisation and Implicit Language Q-Learning.

This class is analogous to the PyTorch Lightning CLI, taking in a DataModule and a LightningModule, 
with some additional functionality and flexibility through using Hydra configurations.

To implement a new Supervised Experiment, you need to make a corresponding LightningModule and DataModule,
and specify their configuration parameters in an associated Hydra configuration file. 

Details of the currently implemented experiments are given below. 

## Premise Selection

The configur

## HOList Training

## Generative Training

### Seq2seq

### Direct Preference Optimisation (DPO)

### Implicit Language Q Learning (ILQL)
