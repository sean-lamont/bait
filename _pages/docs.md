---
permalink: /docs/
title: "Documentation"

layout: single

collection: docs

read_time: false
author_profile: false
share: false
comments: false

classes: wide

sidebar:
    nav: "docs"

---

The design of BAIT is centered around the idea of an 'experiment', which defines a task in ITP (such as premise selection or
proving with an environment).

BAIT is split into the following directories to reflect this: 

### environments
Contains the ITP environments, and code to facilitate the interaction with them.

### models
Contains model architectures, training code and data modules for training.

### experiments
Code for running experiments, which takes a configuration specifying details of the data source, model and possibly environment. 

### configs
Configuration files for running experiments, in Hydra format. 

### data
Includes scripts to download and process raw data from various ITP datasets and benchmarks.

### runs
Contains the output of experiment runs, including log files, proof traces and checkpoints.



# List of datasets
- HOLStep
- LeanStep
- LeanDojo Benchmark Data
- MIZAR40
- HOList Benchmark Data
- HOL4 Premise Selection Data (new)
- HOL4 TacticZero Data

# Environments
- HOList (original)
- Updated HOList (Work in progress)
- LeanDojo (original)
- TacticZero HOL4 Environment (original)
- Updated HOL4 (todo)
- INT

# Models
- ReProver
- HOList Models
- TacticZero
- Embedding Architectures
  - GNN
  - Transformer Encoder
  - Structure Aware Transformers

# Supervised Experiments 
- Premise Selection
  - All embedding architectures
  - HOLStep
  - LeanStep
  - MIZAR40
  - HOL4 Premise Selection Data
   
- HOList Supervised
  - All embedding architectures
  - HOList Benchmark Data

# End-to-End
- TacticZero 
  - All embedding architectures
  - HOL4 TacticZero Data
   
- HOList Eval
  - All embedding architectures
  - HOList benchmark data
   
- End-to-End 
  - LeanDojo
      - Generative (Based on ReProver)
          - Seq2seq training
          - DPO
          - ILQL
      - All search methods
  - HOList 
    - HOList supervised architectures
    - All search methods
     
- Search 
  - HTPS (wip)
  - BestFS
  - Breadth First Search (todo)
  - UpDown (new, wip)
  - Goal scoring models
    - HTPS labelling
    - Polu. labelling

Envs: HOL4, LeanDojo, HOList
Models: All, Generative, Fixed