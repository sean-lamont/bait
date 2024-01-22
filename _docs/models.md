---
permalink: /docs/models/
title: "Models"
---

The Model module includes model architectures.

# Embedding Models
Contains the Embedding architectures currently used for TacticZero, HOList and Premise Selection experiments.

## GNN
Includes the message passing GNN architecture from FormulaNet and HOList, as well as other architectures including
GCN and DiGAE.

## Transformer
Transformer Encoder as described in 

## SAT
Structure Aware Transformer (SAT) models

## Ensemble

# Premise Selection
Contains the Premise Selection model and training code used for Premise Selection experiments.

The forward pass takes a goal and premise as input, embeds them, then concatenates their
embeddings before passing them through a MLP for classification

This model is initialised with embedding architectures for the goal and premise, as well as the architecture for the classifier.

# HOList
## Agent
The code for the live agent used in the HOList Evaluation experiment.

## Supervised
This includes the model and training code for the HOList Supervised experiment. 

# TacticZero
Includes the architecture for the Policy Models used by TacticZero, as well as the original seq2seq based autoencoder.

# End-to-End
Models used for End-to-End experiments are found here. 

## Tactic Models
### ReProver

### Seq2seq

### Direct Preference Optimisation (DPO)

### Implicit Language Q Learning (ILQL)

## Search Models

