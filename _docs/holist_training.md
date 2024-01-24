---
permalink: /docs/holist_training/
title: "HOList Training"
---

This experiment runs the training approach from HOList implemented in `models.HOList.supervised`,
with varying architectures.

# Running
From the root directory of the project simply run:

`python3 -m experiments.supervised_runner --config-name=holist_supervised/{model}`

# Configuration

Configuration files are found in the `configs/premise_selection` directory.
The base configuration file `configs/premise_selection/premise_selection.yaml` specifies the
Premise Selection model and associated DataModule implemented in `models.premise_selection`.

The configuration directory is organised into `{dataset}/{model}`, where each `{dataset}`
includes a base file specifying the vocabulary size, project to log to and the MongoDB database to read from.

Each `{model}` config file contains details of the embedding architecture to use, and the associated `data_type` to inherit from.


# Data
- HOList 

# Models
- Embedding Models 
