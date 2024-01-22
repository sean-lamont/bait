---
permalink: /docs/experiments/
title: "Experiments"
---

Experiments combine Model, Data and possibly Environment modules to run a particular ITP task.
There are intentionally very few constraints on Experiments to allow for a large variety of approaches.

The output of experiments are often used in others. For example, the End-to-End experiment
might use a model checkpoint from a pre-training experiment, and the data generated from the End-to-End run might be used 
in a separate finetuning experiment.





# Configuration

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

Within a config subdirectory, specific datasets and models can be further configured from the base.
For premise selection, we organise this into {dataset}/{model}, whereas other experiments such as TacticZero and HOList
are currently only using one benchmark/dataset, so they are organised based only on {model}. These configurations
inherit from the base experiment, as well as the default model/data configuration in `config/data_type`.
They are the final configuration in the composition order, and are what should be specified when running an experiment.
At a minimum, they should specify the experiment name and model to be run.

# Running

Experiments are run from the root directory,
with the path to the associated configuration file passed in as the `config-name` parameter.
Parameters can be overloaded, added or removed using the Hydra override grammar.


