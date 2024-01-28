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
Includes the message passing GNN architecture from FormulaNet and HOList, as well as other architectures including GCNs.

## Transformer
Standard implementation of the Transformer Encoder.

## SAT
Structure Aware Transformer (SAT) models.

## Ensemble
Ensemble models, which feed the embeddings from a GNN and Transformer model through a final MLP to obtain an aggregated 
final embedding.

# End-to-End
Models used for ~End-to-End experiments are found here. 

## Tactic Models
Models for tactic prediction, map a goal to a set of tactics to run in the environment.

### ReProver 

### Seq2seq

### Direct Preference Optimisation (DPO)

### Implicit Language Q Learning (ILQL)

### HOList Model

## Search Models
Search approaches, selecting one or more goals to work on given a proof state.

### BestFS
### Fringe
### Breadth First Search (BFS)
### HyperTree Proof Search (HTPS)

# HOList
## Agent
The code for the live agent used in the ~HOList Evaluation experiment.

## Supervised
This includes the model and training code for the ~HOList Supervised experiment. 

# Premise Selection
Contains the ~Premise Selection model and training code used for Premise Selection experiments.

The forward pass takes a goal and premise as input, embeds them, then concatenates their
embeddings before passing them through a MLP for classification

This model is initialised with embedding architectures for the goal and premise, as well as the architecture for the classifier.

# TacticZero
Includes the architecture for the Policy Models used by TacticZero, as well as the original seq2seq based autoencoder.
