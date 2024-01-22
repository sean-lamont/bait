---
permalink: /docs/holist/
title: "HOList Evaluation"
---

# Supervised
The supervised HOList experiment follows the model and training procedure of the original paper (link to paper), 
with the model implemented in `models.HOList.supervised`

## Running
From the root directory of the project simply run:

`python3 -m experiments.supervised_runner --config-name=holist_supervised/{model}`


# Evaluation
This experiment is used to evaluate a trained HOList model in the live proving environment.
The code is based on the original DeepMath prover, and is found in `experiments.HOList`

The docker container for the environment must be running as outlined in the Setup documentation.

## Running
To run a HOList evaluation, from the root directory of the project run:

`python3 -m experiments.HOList.holist_eval --config-name=holist_eval/{model}`

There must be a checkpoint file configured which includes the Encoders, Tactic Selection and
Combiner Networks from the HOList Supervised task. The checkpoint file is specified by the
`path_model_prefix` field in `configs/holist_eval/holist_eval.yaml'`, and can be overwritten
from the specific `holist_eval/{model}` file.

The first run of the experiment will generate a checkpoint.npy file in the `theorem_embeddings`
directory specified in the configuration. If the file exists, it will load from the specified directory
