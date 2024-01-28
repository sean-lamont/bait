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
This configuration can inherit configurations from `configs/base`, which define common options such as how directories,
checkpoints and logging are managed.

Within a config subdirectory, specific datasets and models can be further configured from the base.
For premise selection, we organise this into `{dataset}/{model}`, whereas other experiments such as TacticZero and HOList
which use one benchmark/dataset are organised based only on `{model}`. 
These configurations inherit from the base experiment, as well as the default model/data configuration in `config/data_type`.
They are the final configuration in the composition order, and are what should be specified when running an experiment.
At a minimum, they should specify `experiment`, `name` and the model to be run.

# Running

Experiments are run from the root directory,
with the path to the associated configuration file passed in as the `config-name` parameter.
Parameters can be overloaded, added or removed using the Hydra override grammar.

# Lightning Experiments

Many experiments can be fully contained with [PyTorch Lightning](https://lightning.ai/) Modules. In these cases,
the experiment (including logging and checkpoint behaviour) must be specified in the associated LightningModule and LightningDataModule.

The class `experiments.lightning_runner` provides a generic interface for running these experiments.
This class is analogous to the PyTorch Lightning CLI, taking in a DataModule and a LightningModule,
with some additional functionality and flexibility through using Hydra configurations.

Such experiments currently include Premise Selection, the HOList training approach,
and training/fine-tuning generative models.
We also include some currently experimental approaches such as Direct Preference Optimisation and Implicit Language Q-Learning.

To use this class, you need to make a corresponding LightningModule and DataModule,
and specify their configuration parameters in an associated Hydra configuration file. The Hydra file
should specify the LightningModule using the `model` key, with the path to the module in `_target_` and
the parameters listed below this. Similarly for the DataModule, with the `data_module` key.

More complicated experiments require a custom experiment module, and users can refer to the documentation on our
[TacticZero](/bait/docs/tacticzero/) or [End-to-End](/bait/docs/end-to-end) experiments to see some examples.

# Logging
[Weights and Biases](https://wandb.ai/) is the default logging platform used in BAIT, and is automatically integrated into all current experiments.
This can be changed if desired, by modifying the logging source defined in the relevant experiment module.

# Checkpointing
Checkpointing for all experiments which use a LightningModule is easily configured through the associated callbacks
for the trainer in the corresponding `yaml` file.

# Resuming Runs
To resume a run, you should add the following fields to the final configuration file:

```yaml
exp_config.resume: true 
logging_config.id: {wandb_id} #where `wandb_id` is the id associated with the resuming run
exp_config.directory: {base_dir} #where `base_dir` is the root of the directory created from the resuming run.
```


# Sweeps 
Sweeps can be run using the Hydra [multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) functionality.
This allows multiple runs to be set up which vary several configuration keys, and is useful for e.g. hyperparameter sweeping.