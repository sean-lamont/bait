---
permalink: /docs/data/
title: "Data"
---

This page includes details of the datasets included in BAIT and how they are processed.

The `data` directory includes processing scripts and utilities,
separated into each dataset. The processing scripts typically download the raw data files into the 
respective dataset directory, and then process them into the format needed for experiments.


```terminal
├── data
    ├── HOL4
    ├── HOList
    ├── INT
    ├── LeanDojo
    ├── premise_selection
    ├── utils
```

# HOL4

Processed for premise selection and for TacticZero experiments.

By default, creates a `hol4` MongoDB database, with the following collections used by the experiments:
- `split_data`
- `vocab`
- `expression_graphs`
- `paper_goals`

Also includes the following collections with additional data on the HOL4 expressions and their dependencies:
- `dependency_data`
- `expression_metadata`

# HOList

Contains data based on HOL-Light proofs found in the [HOList](https://arxiv.org/pdf/1904.03241v3.pdf) 
benchmark. Used for both the HOList Supervised training experiment,
and for evaluation with the HOList Eval experiment.

By default, creates a `holist` MongoDB database, with the following collections used for the HOList experiments:
- `split_data`
- `vocab`
- `train_thm_ls`
- `expression_graphs`

# INT 
Data processed for the [INT](https://github.com/albertqjiang/INT) environment and experiments.
Uses the INT environment to generate user configured synthetic proving problems, which are saved in this directory.
The processing code for this is taken from the original paper, and unchanged.

# Premise Selection 
## HOLStep
The [HOLStep](https://arxiv.org/abs/1703.00426) premise selection dataset.

Processed for Premise Selection experiments.
Note that full graph processing takes a very long time, with over 1M expressions.

By default, creates a `holstep` MongoDB database, with the following collections used for Premise Selection:
- `split_data`
- `vocab`
- `expression_graphs`
 
## LeanStep

Data from the [LeanStep](https://github.com/jesse-michael-han/lean-step-public) benchmark,
processed for Premise Selection experiments. Modified to include s-expression output to enable graph parsing.

By default, creates a `leanstep` MongoDB database, with the following collections for premise selection:
- `split_data`
- `vocab`
- `expression_graphs`


Also processes and stores additional co-training data as outlined in the paper, for possible future experiments,
with the following collections:
- `next_lemma`
- `premise_classification`
- `proof_step`
- `proof_term`
- `skip_proof`
- `theorem_name`
- `type_prediction`
 
## MIZAR40

The [MIZAR40](https://github.com/JUrban/deepmath) premise selection dataset. Processed for the format in Premise Selection experiments.

By default, creates a `mizar40` MongoDB database, with the following collections used for Premise Selection:
- `split_data`
- `vocab`
- `expression_graphs`

## LeanDojo

Data generated from the [LeanDojo](https://github.com/lean-dojo/LeanDojo) benchmark, 
used for End-to-End experiments and for training ReProver based models.
Processed into a format which enables training generative models.