---
permalink: /docs/end-to-end/
title: "End-to-End Proving (Under Development)"
---


**Currently under development, this documentation will be continually updated.**

The End-to-End experiment aims to provide an implementation of the
abstract [AI-ITP setup.](https://sean-lamont.github.io/bait/#system-overview)

It is designed to be modular, allowing for different combinations of Tactic and Search models to be run on
different Environments with minimal effort.

An End-to-End experiment takes as input the Tactic Model, Search Model and Environment specified in a Hydra config file.
It runs the specified model on proofs in the environment, collecting the proof logs.
Following this, it can take other user specified experiments to run.
These experiments are expected to use the newly generated data to train the Tactic and Search models.
The newly trained models are then loaded and the process is repeated.

[//]: # (Originally based on [ReProver]&#40;https://github.com/lean-dojo/ReProver&#41;, this experiment allows for synthetic data to be)

[//]: # (incorporated into training agents.)

It provides the following features:

- Abstract interface, allows for different Environments, Tactic and Search models to be combined with minimal effort
- Tactic, Search and Environment interfaces are modular and can be extended easily
- Distributed evaluation and training, adapting to the resources available (as specified in the configuration file)
- Automated logging with WandB
- Automatically generates proof traces in a standardised format
- Interactive visualisation of proof attempts

Given the large scope of the project, there may be some bugs or incomplete features.
We are working to address these, as well as integrate new Tactic, Search and Environment modules over time.

That being said, the current implementation has proven to be effective in training and evaluating several Tactic and
Search approaches across different Environments. It is also the only fully open source implementation of the AI-ITP
loop to date.

The current set of combinations of Models and Environments which can be run are summarised below. Items which are
bolded have not yet been extensively tested.

## Search Models

- Best First Search (BestFS)
- **HyperTree Proof Search (HTPS)**
- Breadth First Search (BFS)
- Fringe

## Environments

- LeanDojo
- HOList
- HOL4

## Tactic Models

### Generative models based on ReProver

Pre-trained models based on ReProver, extended to support fine-tuning with synthetic data.

- Currently only trained/tested with LeanDojo Environment

- Fine-tuning with the following approaches:
    - seq2seq
    - **DPO (Direct Preference Optimisation)**
    - **ILQL (Implicit Lanuage Q-Learning)**

### Original HOList Tactic Models

- Supports varying embedding architectures (GNN, Transformer, SAT, Directed SAT, Ensemble)
- Currently only tested on the HOList environment, using the fixed tactic set from the original HOList
as well as s-expression parsing

### TacticZero based model

- Incorporates the TacticZero tactic/argument generation architecture.
- Only tested so far with HOL4, using the fixed tactic set from the original TacticZero.
- Currently only implemented for evaluation, training is not yet supported.
    - To add this, logs from the proof search would need to be processed for policy gradient (if replicating the
      original TacticZero approach),
      or to a new approach (e.g. supervised training)

# Some possible future additions

### LeanDojo

- Add tactic models which are restricted to a small/fixed subset as done in HOList or TacticZero models.

### HOList

- Generative ReProver style model (currently would be restricted by the environment, as outlined in the ~HOList docs)

### HOL4

- Generative ReProver style model
- TacticZero training

# Modules

## end_to_end_experiment

The module for running the experiment. Takes a configuration file specifying the tactic model,
search model, environment, how to process traces for model training, and what modules to call for training.

## proof_node

Implements the Proof Search Tree datastructure.

## search_result

Class which contains a SearchResult object, which includes all relevant information from a proof search

## visualise_trace

Allows for an interactive visualisation of the proof search.
Requires separate implementations for each new search

## Tactic/Search models with Lightning

Both tactic and search models are assumed to be Lightning Modules.

For training, they should each have an associated DataModule.
This should define how to process proof traces before training. For
example, `models.end_to_end.search_models.goal_model`
labels all proven nodes as 1, and all unproven goals over a visit count threshold as 0 (used in HTPS).
`models.end_to_end.tactic_models.dpo` implements Direct Preference Optimisation, ranking edges based on errors and
proofs.
Tactic models need to implement a get_tactic method which maps a string to a tactic.
Aside from this, the models have no restrictions.
Current models include HOList Tactic Generator, generative models with Seq2Seq training, DPO and ILQL training.
Once these are implemented, they can be added to tac_models or search_models
respectively.

# Running

# Configuration

# Examples

The below configurations are some examples of End-to-End experiments:

## ReProver

- Run original model trained on the LeanDojo benchmark with BestFS
- BestFS, updating tactic with synthetic data using seq2seq training
- BestFS, updating tactic with synthetic data using DPO training
- BestFS, updating tactics with synthetic data using ILQL training
- HTPS, updating tactics with synthetic data using seq2seq training, and updating HTPS goal model

## HOList

- Run original HOList model for training and evaluation, using new abstract and shared components

## HOL4

- Run TacticZero in evaluation mode, with any search strategy 

