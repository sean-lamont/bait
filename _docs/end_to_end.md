---
permalink: /docs/end-to-end/
title: "End-to-End Experiments"
---

## TacticZero

## HOList Evaluation



## Abstract End-to-End Loop

A central feature of BAIT is the abstraction over core components utilised across many ITP automation methods.
Implementing this functionality are the following modules as part of the end_to_end experiment:

- `proof_node`
- `search_models`
- `search_result`
- `tac_models`

The `end_to_end_experiment` module links these together. A configuration file specifying the tactic model,
search model, environment, how to process traces for model training, and what modules to call for training.

## proof_node

Implements the Proof Search Tree datastructure.

## search_models

Implements abstract and concrete search models.

## tac_models

Implements abstract and concrete tactic selection models

## search_result

Class which contains a SearchResult object, which includes all relevant information from a proof search

## visualise_trace

Allows for an interactive visualisation of the proof search.
Requires separate implementations for each new search

## Tactic/Search models with Lightning

Both tactic and search models are lightning modules.

They should each have an associated DataModule.
This should define how to process proof traces before training. For example, goal_model takes
all proven nodes as a 1, and all unproven goals over a visit count as 0. DPO ranks edges based on errors,
and generator just takes a seq2seq loss over proven goals.

Tactic models need to implement a get_tactic method which maps a string to a tactic.
Aside from this, the models have no restrictions.
Current models include HOList Tactic Generator, generative models with Seq2Seq training, DPO and ILQL training,
Goal models with varying objectives etc. Once these are implemented, they can be added to tac_models or search_models
respectively.
