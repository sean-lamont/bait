---
permalink: /docs/premise_selection/
title: "Premise Selection"
---

Premise Selection is a common benchmark for approaches in AI-ITP. 

# Running 
To run a premise selection experiment, from the root directory of the project simply run:

`python3 -m experiments.supervised_runner --config-name=premise_selection/{dataset}/{model}`

where `{dataset}` is the desired dataset, and `{model}` is the desired model.
To change model hyperparameters, modify the appropriate `{dataset}/{model}` config file.

# Configuration

Configuration files are found in the `configs/premise_selection` directory.
The base configuration file `configs/premise_selection/premise_selection.yaml` specifies the 
Premise Selection model and associated DataModule (only one at present). 

The configuration directory is organised into `{dataset}/{model}`, where each `{dataset}` 
includes a base file specifying the vocabulary size, project to log to and the MongoDB database to read from.

Each `{model}` config file contains details of the embedding architecture to use, and the associated `data_type` to inherit from.