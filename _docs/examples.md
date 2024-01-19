---
permalink: /docs/examples/
title: "Examples"
---

## Examples
### Premise Selection
To run a premise selection experiment, from the root directory of the project simply run:

`python3 -m experiments.premise_selection --config-name=premise_selection/{dataset}/{model}`

where {dataset} is the desired dataset, and {model} is the desired model.
To change model hyperparameters, modify the appropriate {dataset}/{model} config file.

### HOList Supervised
To run a premise selection experiment, from the root directory of the project simply run:

`python3 -m experiments.holist_supervised --config-name=holist_supervised/{model}`

### HOList Evaluation
To run a HOList evaluation, from the root directory of the project run:

`python3 -m experiments.holist_eval --config-name=holist_eval/{model}`

There must be a checkpoint file configured which includes the Encoders, Tactic Selection and
Combiner Networks from the HOList Supervised task. The checkpoint file is specified by the
`path_model_prefix` field in `configs/experiments/holist_eval/holist_eval.yaml'`, and can be overwritten
from the specific `holist_eval/{model}` file.

The default value, where you can copy HOList supervised checkpoints to is:

`path_model_prefix: 'experiments/holist/checkpoints/checkpoint'`

The first run of the experiment will generate a checkpoint.npy file in the `theorem_embeddings`
directory specified in the configuration. If the file exists, it will load from the specified directory.

### TacticZero
To run a TacticZero experiment, from the root directory of the project simply run:

`python3 -m experiments.tacticzero_experiment --config-name=tacticzero/{model}`

## Resuming Runs
To resume a run, you should add the following fields to the final configuration file:

- `exp_config.resume: True`
- `logging_config.id: {wandb_id}` where `wandb_id` is the id associated with the resuming run
- `exp_config.directory: {base_dir}` where `base_dir` is the root of the directory created from the resuming run.
  By default, this is in the format:
