---
permalink: /docs/supervised/
title: "Supervised Experiments"
---

# Supervised

Many experiments naturally conform to what we call a 'Supervised' pattern. In these cases, 
the experiment must be fully captured in a LightningModule and a LightningDataModule 
without the need for custom checkpointing or logging behaviour. 

The class `experiments.supervised_runner` provides a generic interface for running these experiments.

Such experiments currently include Premise Selection, the HOList training approach, 
and training/fine-tuning generative models (based on ReProver).
We also include some currently experimental approaches such as Direct Preference Optimisation and Implicit Language Q-Learning. 

This class is analogous to the PyTorch Lightning CLI, taking in a DataModule and a LightningModule, 
with some additional functionality and flexibility through using Hydra configurations.

To use this class, you need to make a corresponding LightningModule and DataModule,
and specify their configuration parameters in an associated Hydra configuration file. The Hydra file
should specify the LightningModule using the `experiment` key, with the path to the module in `_target_` and
the parameters listed below this. Similarly for the DataModule, with the `data_module` key. 

More complicated experiments require a custom experiment module, and users can refer to the documentation on our 
TacticZero or End-to-End experiments to see some examples.  

## Premise Selection

The configur

## HOList Training

## Generative Training

### Seq2seq

### Direct Preference Optimisation (DPO)

### Implicit Language Q Learning (ILQL)
