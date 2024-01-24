---
permalink: /docs/data/
title: "Data"
---

This page includes details of the datasets included in BAIT, and how they are processed.

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


# HOLStep

Processed for Premise Selection experiments.
Note that full graph processing takes a very long time, with over 1M expressions.

By default, creates a `holstep` MongoDB database, with the following collections used for Premise Selection:
- `split_data`
- `vocab`
- `expression_graphs`
 
# LeanStep

Processed for Premise Selection experiments. Modified to include s-expression output to enable graph parsing.

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
 
# MIZAR40

Processed for Premise Selection experiments.

By default, creates a `mizar40` MongoDB database, with the following collections used for Premise Selection:
- `split_data`
- `vocab`
- `expression_graphs`

# HOList

Contains data based on HOL-Light proofs. Used for both the HOList Supervised training experiment,
and for evaluation with the HOList Eval experiment.

By default, creates a `holist` MongoDB database, with the following collections used for the HOList experiments:
- `split_data`
- `vocab`
- `train_thm_ls`
- `expression_graphs`

## LeanDojo

Data generated from the LeanDojo benchmark. Processed into a format which enables training generative models.


[//]: # (## Premise Selection)

[//]: # ()

[//]: # (Premise selection data is processed to keep a consistent format between datasets.)

[//]: # (In this way, the data source can be abstracted from the experiments which allows for)

[//]: # (datasets and models to be swapped with minimal additional code.)
