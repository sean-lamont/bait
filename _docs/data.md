---
permalink: /docs/environments/
title: "Data"
---

This page includes details of the datasets included in BAIT, and how they are processed.

## Processing
Most data in BAIT is processed to use MongoDB.

Datasets relating to Embedding Experiments are processed to include attributes relevant for
both graph and sequence based architectures. The datasets which this currently relates to are 


## Premise Selection
Premise selection data is processed to keep a consistent format between datasets. 
In this way, the data source can be abstracted from the experiments which allows for 
datasets and models to be swapped with minimal additional code.

### HOLStep
### LeanStep
### MIZAR40
### HOL4 

## HOList
Contains data based on HOL-Light proofs. Used for both the HOList Supervised training experiment,
and for evaluation with the HOList Eval experiment.

## LeanDojo 
Data generated from the LeanDojo benchmark. Processed into a format which enables training generative models.



