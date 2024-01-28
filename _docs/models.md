---
permalink: /docs/models/
title: "Models"
---

The model module contains model architectures, training code and data modules for training.
Models are written in PyTorch, with training and data loading/processing typically using 
PyTorch Lightning. The directory structure is listed below.

```terminal
├── models
    ├── embedding_models
    ├── end_to_end
    ├── HOList
    ├── INT
    ├── premise_selection
    ├── TacticZero
```


# Embedding Models
Contains the Embedding architectures currently used for TacticZero, HOList and Premise Selection experiments.

## GNN
Includes the message passing GNN architecture from [FormulaNet](https://arxiv.org/abs/1709.09994) and
[HOList](/bait/docs/), as well as other architectures including GCNs.

## Transformer
Standard implementation of the [Transformer Encoder](https://arxiv.org/abs/1706.03762).

## SAT
[Structure Aware Transformer](https://arxiv.org/abs/2202.03036)(SAT), including the [Directed](https://arxiv.org/abs/2210.13148) variant.

## Ensemble
Ensemble models, which feed the embeddings from a GNN and Transformer model through a final MLP to obtain an aggregated 
final embedding.

# End-to-End
Models used for [End-to-End experiments](/bait/docs/end-to-end) are found here. 

## Tactic Models (todo)
Models for tactic prediction, map a goal to a set of tactics to run in the environment.

### ReProver 

### Seq2seq

### Direct Preference Optimisation (DPO)

### Implicit Language Q Learning (ILQL)

### HOList Model

## Search Models (todo)
Search approaches, selecting one or more goals to work on given a proof state.

### BestFS
### Fringe
### Breadth First Search (BFS)
### HyperTree Proof Search (HTPS)

# HOList
## Agent
The code for the live agent used in the [HOList Evaluation](/bait/docs/holist) experiment.

## Supervised
This includes the model and training code for the HOList Supervised experiment, with the GNN architecture used in
[this paper](https://arxiv.org/pdf/1905.10006.pdf).

# INT
The models used for the [INT experiments](/bait/docs/int). Includes GNN and Transformer Encoders, as defined in `models.INT`.

# Premise Selection
Contains the [Premise Selection](/bait/docs/premise_selection/) model and training code used for Premise Selection experiments.

The forward pass takes a goal and premise as input, embeds them, then concatenates their
embeddings before passing them through a MLP for classification

This model is initialised with embedding architectures for the goal and premise, as well as the architecture for the classifier.

# TacticZero
Includes the architecture for the Policy Models used in [TacticZero experiments](/bait/docs/tacticzero), as well as the original seq2seq based autoencoder.
