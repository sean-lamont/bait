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

The design of BAIT is centered around the idea of an *experiment*, which defines a task in ITP such as premise
selection or proving with an environment.


BAIT is split into the following directories to reflect this, which we expand on below:

```terminal
├── bait
    ├── configs
    ├── data
    ├── environments
    ├── experiments
    ├── models
    ├── runs
```


## environments

Contains the ITP environments, and code to facilitate interfacing with them.

## models

Contains model architectures, training code and data modules for training.
Models are written in PyTorch, with training and data loading/processing using 
PyTorch Lightning.

## experiments

Code for running experiments.
Experiments take a Hydra configuration,
generally specifying details of the relevant data source, model and possibly environment.

## configs

Configuration files for running experiments, in Hydra format.

## data

Includes scripts to download and process raw data from various ITP datasets and benchmarks.

## runs

Contains the output of experiment runs, including log files, proof traces and checkpoints.