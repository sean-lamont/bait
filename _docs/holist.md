---
permalink: /docs/holist/
title: "HOList Evaluation"
---

This experiment is used to evaluate a HOList model in the corresponding live proving environment.
The code is based on the original DeepMath prover, and is found in `experiments.HOList`

The docker container for the HOList environment must be running as outlined in the relevant [documentation](/bait/docs/setup/#holist).

## Running
To run a HOList evaluation, from the root directory of the project run:

`python3 -m experiments.HOList.holist_eval --config-name=holist_eval/{model}`

There must be a checkpoint file configured which includes the Encoders, Tactic Selection and
Combiner Networks from the HOList Supervised task. The checkpoint file is specified by the
`path_model_prefix` field in `configs/holist_eval/holist_eval.yaml'`, and can be overwritten
from the specific `holist_eval/{model}` file.

The first run of the experiment will generate a checkpoint.npy file in the `theorem_embeddings`
directory specified in the configuration. If the file exists, it will load from the specified directory

# Data
- [HOList](/bait/docs/data/#holist)

# Models
- [Embedding Architectures](/bait/docs/models/#embedding-models)
- [HOList Agent](https://sean-lamont.github.io/bait/docs/models/#agent)

# Environments 
- [HOList](/bait/docs/environments/#holist)