---
permalink: /docs/experiments/
title: "Experiments"
---

Experiments are intentionally abstract.

They combine Model, Data and possibly Environment modules to run a particular ITP task.

This includes: 
- Pretraining
- Finetuning
- Premise Selection
- Running End-to-End provers

The output of Experiments are often used in other Experiments. For example, the End-to-End Experiment
can use a model checkpoint from a pre-training Experiment, then the data generated can be used 
in a finetuning Experiment. Or the HOL4 premise selection Experiment can be run to generate checkpoints to use in the
TacticZero experiment.


## Configuration

We use Hydra as our configuration management library. This allows for flexible, minimal changes
to configuration files for running experiments. There are several 'levels' of hydra configuration which
are analogous to class inheritance.

Specific experiments should include a subfolder in the `configs` directory,
such as `premise_selection`. In the root of the subfolder, you can implement a configuration
file as a base for the experiment, with default configurations for the specific experiment.
For example, `configs/tacticzero/tactic_zero.yaml` defines the specific tactics used
in TacticZero, as well as default values for the number of steps `max_steps`, number of epochs etc.
This configuration should inherit some or all of the configurations under `configs/base`, which define how directories,
checkpoints and logging are managed.

[//]: # (todo details of the configurations for each experiment can be found in their respective documentation )

Within a config subdirectory, specific datasets and models can be configured from the base.
For premise selection, this is organised into {dataset}/{model}, whereas other experiments such as TacticZero and HOList
are
currently only using one benchmark/dataset, so they are organised based only on {model}. These configurations
inherit from the base experiment, as well as the default model/data configuration in `config/data_type`.
They are the final configuration in the composition order, and are what should be specified when running an experiment.
At a minimum, they should specify the experiment name and model to be run.

## Running

Experiments are run modules from the root directory,
with the path to the associated configuration file passed in as the `config-name` parameter.
Parameters can be overloaded, added or removed using the Hydra override grammar.