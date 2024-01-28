---
permalink: /docs/tacticzero/
title: "TacticZero Experiments"
---

TacticZero experiments use the HOL4 environment and dataset as outlined in the original [paper](https://arxiv.org/abs/2102.09756). 
It is an online policy gradient based algorithm. The architectures are defined in `models.TacticZero`, with the
agent interaction loop being defined in `experiments.TacticZero`.


# Running
To run a TacticZero experiment, from the root directory of the project simply run:

```terminal
python3 -m experiments.TacticZero.tacticzero_experiment --config-name=tacticzero/{model}
```

# Configuration

Users can specify what architectures to use for the goal, tactic and argument selection modules through the
configurations in `configs/tacticzero`. Here, `configs/tacticzero/tactic_zero` is the base config, and the other config files
are used to define model details for the GNN, Transformer and original Autoencoder used.
Pretrained embedding models from Premise Selection experiments can be included by setting `pretrain: true` and `pretrain_ckpt`
to the path of the pretrained checkpoint.

# Data 
- [HOL4 Data](/bait/docs/data/#hol4)

This experiment assumes data has been processed into a MongoDB database, with the following yaml keys:

```yaml
# MongoDB database (default is 'hol4')
data_config.data_options.db
# Processed Data for each expression/formula. Default is 'expression_graphs'
data_config.data_options.expression_col 
# Collection mapping each token to an index. Default is 'vocab_col'
data_config.data_options.vocab_col 
# Collection containing the train/val/test split of the goals to prove
data_config.data_options.paper_goals
```

# Environments
- [HOL4](/bait/docs/environments/#hol4)
 
# Models
- [Embedding Architectures](/bait/docs/models/#embedding-models)
- [TacticZero Model](/bait/docs/models/#tacticzero)