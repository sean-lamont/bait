---
permalink: /docs/end-to-end/
title: "End-to-End Experiments"
---

## Abstract End-to-End Loop (work in progress)
The End-to-End experiment aims to implement the AI-ITP loop directly in a general way, 
where the user only needs to specify the Model, Search and Environment with minimal additional boilerplate.

Based on the 


It currently includes the LeanDojo environment with generative models based on ReProver,
and an updated HOList environment with the associated tactic models used in the paper. 

Tactic Models:
- LeanDojo
    - ReProver 
    - DPO 
    - ILQL
- HOList 
  - HOList Tactic Models, with varying embedding architectures

Search Models:
- HTPS
- UpDown
- Best First Search (BestFS)
- Breadth First Search (BFS)
- Fringe 

Environments:
- LeanDojo
- HOList
- HOL4 (in progress)


The proof search and logs are tracked in a standard format, and will automatically detect 
proofs and reconstruct them from the search. Proof search can be visualised, with the chosen tactics,
model scores and environment response.


A central feature of BAIT is the abstraction over core components utilised across many ITP automation methods.
Implementing this functionality are the following modules as part of the end_to_end experiment:

The `end_to_end_experiment` module links these together. A configuration file specifying the tactic model,
search model, environment, how to process traces for model training, and what modules to call for training.

## proof_node

Implements the Proof Search Tree datastructure.

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
