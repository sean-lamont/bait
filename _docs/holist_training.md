---
permalink: /docs/holist_training/
title: "HOList Training"
---

This experiment runs the training approach from HOList implemented in `models.HOList.supervised`,
with varying architectures.

# Running
From the root directory of the project simply run:

```terminal
python3 -m experiments.lightning_runner --config-name=holist_supervised/{model}
```

# Configuration

Configuration files are found in the `configs/holist_supervised` directory.
The base configuration file in `configs/premise_selection/holist_supervised.yaml` specifies the
training model and associated DataModule implemented in `models.HOList.supervised`.

Each `{model}` config file contains details of the embedding architecture to use, and the associated `data_type` to inherit from.


# Data
- [HOList](/bait/docs/data/#holist)
 
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
- [Embedding Architectures](/bait/docs/models/#embedding-models)
- [HOList Supervised Model](/bait/docs/models/#supervised)
