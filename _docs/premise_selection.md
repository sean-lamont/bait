---
permalink: /docs/premise_selection/
title: "Premise Selection"
---

Premise Selection is a common benchmark for approaches in AI-ITP. 

# Running 
To run a premise selection experiment, from the root directory of the project simply run:

```terminal
python3 -m experiments.lightning_runner --config-name=premise_selection/{dataset}/{model}
```

where `{dataset}` is the desired dataset, and `{model}` is the desired model.
To change model hyperparameters, modify the appropriate `{dataset}/{model}` config file.

These experiments require the appropriate dataset has been processed as outlined in the setup page.

# Configuration

Configuration files are found in the `configs/premise_selection` directory.
The base configuration file `configs/premise_selection/premise_selection.yaml` specifies the 
Premise Selection model and associated DataModule implemented in `models.premise_selection`. 

The configuration directory is organised into `{dataset}/{model}`, where each `{dataset}` 
includes a base file specifying the vocabulary size, project to log to and the MongoDB database to read from.

Each `{model}` config file contains details of the embedding architecture to use, and the associated `data_type` to inherit from.

# Data
- [HOLStep](/bait/docs/data/#holstep)
- [LeanStep](/bait/docs/data/#leanstep)
- [MIZAR40](/bait/docs/data/#mizar40)
- [HOL4 Dataset](/bait/docs/data/#hol4)

## Processing
This experiment assumes data has been processed into a MongoDB database, with the following yaml keys:

```yaml
# MongoDB database (defaults are 'hol4', 'mizar40', 'leanstep', 'holstep')
data_config.data_options.db
# Processed Data for each expression/formula. Default is 'expression_graphs'
data_config.data_options.expression_col 
# Collection mapping each token to an index. Default is 'vocab_col'
data_config.data_options.vocab_col 
# Collection containing the train/val/test split of the data. Default is 'split_data'
data_config.data_options.split_col 
```

# Models
- [Embedding Architectures](/bait/docs/models/#embedding-models)
- [Premise Selection](/bait/docs/models/#premise-selection)
