---
permalink: /docs/end-to-end/
title: "End-to-End Proving (Under Development)"
---

The End-to-End experiment aims to provide an implementation of the abstract [AI-ITP setup.](https://sean-lamont.github.io/bait/#system-overview)

Currently under development, the documentation will be updated in a future release.

[//]: # (The experiment takes as input the Tactic Model, Search Model and Environment.)

[//]: # (It runs the specified model on proofs in the environment, collecting the proof logs.)

[//]: # (Following this, it can take other user specified experiments to run)

[//]: # (with the newly generated data, which are expected to train the Tactic and Search models.)

[//]: # (The newly trained models are then loaded and the process is repeated.)

[//]: # ()
[//]: # (Originally based on [ReProver]&#40;https://github.com/lean-dojo/ReProver&#41;, this experiment allows for synthetic data to be incorporated into training agents.)

[//]: # ()
[//]: # (It provides the following features:)

[//]: # ()
[//]: # (- Abstract interface, allows for different Environments, Tactic and Search models to be combined with minimal effort)

[//]: # (- Distributed evaluation, which is adaptable to the resources available &#40;specified in the configuration file&#41;)

[//]: # (- Automated logging with WandB)

[//]: # (- Automatically generates proof traces in a standardised format)

[//]: # (- Proof attempts can be visualised with `visualise_trace.py`)

[//]: # ()
[//]: # (The current set of combinations of Models and Environments which can be run are summarised below.)

[//]: # ()
[//]: # (It currently includes the LeanDojo environment with generative models based on ReProver,)

[//]: # (and an updated HOList environment with the associated tactic models used in the paper.)

[//]: # ()
[//]: # (# Working)

[//]: # ()
[//]: # (## Search Models)

[//]: # (- Best First Search &#40;BestFS&#41;)

[//]: # ()
[//]: # (## LeanDojo Environment)

[//]: # ()
[//]: # (## Tactic Models)

[//]: # (- Generative models based on ReProver &#40;currently only trained/tested with LeanDojo Environment&#41;)

[//]: # (- Trained with the following approaches:)

[//]: # (    - seq2seq)

[//]: # (    - DPO)

[//]: # (    - ILQL)

[//]: # ()
[//]: # (- HOList Tactic Models, with varying embedding architectures)

[//]: # (  - Will only work with HOList environment, as they are tailored for this)

[//]: # ()
[//]: # (## HOList Environment)

[//]: # ()
[//]: # (# Work in progress)

[//]: # ()
[//]: # (## Search Models)

[//]: # (- HTPS)

[//]: # (- UpDown)

[//]: # (- Breadth First Search &#40;BFS&#41;)

[//]: # (- Fringe)

[//]: # ()
[//]: # (# Possible future additions)

[//]: # ()
[//]: # (### LeanDojo)

[//]: # ()
[//]: # (- Add tactic models which are restricted to a small subset as done in HOList or HOL4)

[//]: # ()
[//]: # (### HOList)

[//]: # ()
[//]: # (- Generative model &#40;currently would be restricted by the enviroment, as outlined in the ~HOList docs&#41;)

[//]: # ()
[//]: # ()
[//]: # (# Modules)

[//]: # (## end_to_end_experiment)

[//]: # (The module for running the experiment. Takes a configuration file specifying the tactic model,)

[//]: # (search model, environment, how to process traces for model training, and what modules to call for training.)

[//]: # ()
[//]: # (## proof_node)

[//]: # ()
[//]: # (Implements the Proof Search Tree datastructure.)

[//]: # ()
[//]: # (## search_result)

[//]: # ()
[//]: # (Class which contains a SearchResult object, which includes all relevant information from a proof search)

[//]: # ()
[//]: # (## visualise_trace)

[//]: # ()
[//]: # (Allows for an interactive visualisation of the proof search.)

[//]: # (Requires separate implementations for each new search)

[//]: # ()
[//]: # (## Tactic/Search models with Lightning)

[//]: # ()
[//]: # (Both tactic and search models are assumed to be Lightning Modules.)

[//]: # ()
[//]: # (For training, they should each have an associated DataModule.)

[//]: # (This should define how to process proof traces before training. For example, `models.end_to_end.search_models.goal_model` )

[//]: # (labels all proven nodes as 1, and all unproven goals over a visit count threshold as 0 &#40;used in HTPS&#41;.)

[//]: # (`models.end_to_end.tactic_models.dpo` implements Direct Preference Optimisation, ranking edges based on errors and proofs.)

[//]: # ()
[//]: # (Tactic models need to implement a get_tactic method which maps a string to a tactic.)

[//]: # (Aside from this, the models have no restrictions.)

[//]: # (Current models include HOList Tactic Generator, generative models with Seq2Seq training, DPO and ILQL training. )

[//]: # (Once these are implemented, they can be added to tac_models or search_models)

[//]: # (respectively.)

[//]: # ()
[//]: # (# Running) 

[//]: # ()
[//]: # (# Configuration )

[//]: # ()
[//]: # (# Examples )

[//]: # ()
[//]: # (The below configurations are some examples of End-to-End experiments:)

[//]: # ()
[//]: # (## ReProver)

[//]: # ()
[//]: # (- Run original model trained on the LeanDojo benchmark with BestFS)

[//]: # (- BestFS, updating tactic with synthetic data using seq2seq training)

[//]: # (- BestFS, updating tactic with synthetic data using DPO training)

[//]: # (- BestFS, updating tactics with synthetic data using ILQL training)

[//]: # (- HTPS, updating tactics with synthetic data using seq2seq training, and updating HTPS goal model  )

[//]: # ()
[//]: # (## HOList)

[//]: # (- Run original HOList model, using the new abstract and shared components)
