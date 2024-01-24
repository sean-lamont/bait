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

## Processing
This experiment assumes data has been processed into a MongoDB database, with the following yaml keys:

```yaml
# MongoDB database (default is 'holist')
data_config.data_options.db
# Processed Data for each expression/formula. Default is 'expression_graphs'
data_config.data_options.expression_col 
# Collection mapping each token to an index. Default is 'vocab_col'
data_config.data_options.vocab_col 
# Collection containing the train/val/test split of the data. Default is 'split_data'
data_config.data_options.split_col
# Collection containing the list of theorems, used to randomly sample negative premises
data_config.data_options.thms_col 
```


# Models
- Embedding Models 
