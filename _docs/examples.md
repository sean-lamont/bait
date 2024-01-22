---
permalink: /docs/examples/
title: "Examples"
---

# Premise Selection
To run a premise selection experiment, from the root directory of the project simply run:

`python3 -m experiments.supervised_runner --config-name=premise_selection/{dataset}/{model}`

where {dataset} is the desired dataset, and {model} is the desired model.
To change model hyperparameters, modify the appropriate {dataset}/{model} config file.

# HOList

## Supervised
To run the HOList training experiment, from the root directory of the project simply run:

`python3 -m experiments.supervised_runner --config-name=holist_supervised/{model}`

## Evaluation
To run a HOList evaluation, from the root directory of the project run:

`python3 -m experiments.HOList.holist_eval --config-name=holist_eval/{model}`

There must be a checkpoint file configured which includes the Encoders, Tactic Selection and
Combiner Networks from the HOList Supervised task. The checkpoint file is specified by the
`path_model_prefix` field in `configs/holist_eval/holist_eval.yaml'`, and can be overwritten
from the specific `holist_eval/{model}` file.

The first run of the experiment will generate a checkpoint.npy file in the `theorem_embeddings`
directory specified in the configuration. If the file exists, it will load from the specified directory

# TacticZero
To run a TacticZero experiment, from the root directory of the project simply run:

`python3 -m experiments.TacticZero.tacticzero_experiment --config-name=tacticzero/{model}`