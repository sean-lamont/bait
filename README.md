# Setup Instructions

## Python packages
### Install torch, torch_geometric and PyG supporting packages based on CUDA version (11.7 shown here)
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
- pip install torch_geometric 
- pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

### Run requirements.txt
- pip install -r requirements.txt
 
### Install remaining packages 
- pip install pyfarmhash
- pip install einops matplotlib plotly igraph pymongo pyrallis wandb dill pyfarmhash absl-py grpcio-tools pexpect torchtext

## Setup Data
You will need a working MongoDB server. To install one locally, you can follow the instructions based on your OS,
with e.g. Ubuntu instructions available here: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/

### Download database dump
We provide preprocessed databases containing data necessary for experiments across all platforms. You can download these at (removed for anonymity)

You can save a current mongodb file for restoration with: 

`mongodump --archive={file}.gz --gzip`

Once you have the archive installed, you can run `mongorestore --gzip --archive={archive}` where `archive` is the download.


## Setup

### HOL4
#### Environment
- Download and install polyml: https://polyml.org/download.html 
- Run the HOL4 build script `bash environments/hol4/build_hol.sh`
#### Data
- Raw data included in repository 
- Run `python -m data.hol4.process_hol4` from the root project directory

### LeanStep
Install Lean3:

- Globally
  - wget -q https://raw.githubusercontent.com/leanprover-community/mathlib-tools/master/scripts/install_debian.sh && bash install_debian.sh ; rm -f install_debian.sh && source ~/.profile 
- (preferred) Inside venv:
  - curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
  - source $HOME/.elan/env to add elan to path, or copy/symlink $HOME/.elan/env to venv bin folder
  - pip install mathlibtools

- Run `bash data/lean/lean-step/setup.sh` from project root (as this is a large dataset, you can
 terminate the parallel_data_gen script early and run the final line for a reduced dataset)
- Run `python -m data.lean.process_leanstep`
 

### HOList
#### Environment
- Setup the Docker container which provides an API to a custom HOL-Light environment setup by Google for AITP.
- Run `bash environments/holist/setup_hollight.sh`
- (Obtained from https://github.com/brain-research/hol-light.git, with licence in directory)

#### Data
- obtain data from https://storage.googleapis.com/deepmath/deephol.zip and place it in a `raw_data` directory in `data/holist`
- Run `python -m data.holist.process_holist`


### MIZAR
- Run `bash data/mizar/get_mizar.sh`
- Run `python -m data.mizar.process_mizar`
- 
### HOLStep
- Run `bash data/holstep/get_holstep.sh`
- Run `python -m data.holstep.process_holstep`

[//]: # (### INT)

[//]: # (sudo apt-get install libopenmpi-dev)

[//]: # (pip install baselines )

[//]: # (pip install git+https://github.com/openai/baselines@ea25b9e8)

[//]: # ()


# Running Experiments

## Configs
We use Hydra as our configuration management library. This allows for flexible, minimal changes 
to configuration files for running experiments. There are several 'levels' of hydra configuration which 
are analogous to class inheritance. 

Specific experiments should include a subfolder in the `config/experiments` directory,
such as `premise_selection`. In the root of the subfolder, they should implement a configuration
file as a base for the experiment, with default configurations for the specific experiment.
For example, `config/experiments/tacticzero/tactic_zero.yaml` defines the specific tactics used
in TacticZero, as well as default values for the number of steps `max_steps`, number of epochs etc.
 This configuration
should inherit some or all of the configurations under `config/base`, which define how directories,
checkpoints and logging 
is managed, as well as the data source.

Within an experiment subdirectory, specific datasets and models can now be configured from the base.
For premise selection, this is organised into {dataset}/{model}, whereas other experiments such as TacticZero and HOList are
currently only using one benchmark/dataset, so they are organised based only on {model}. These configurations
inherit from the base experiment, as well as the default model/data configuration in `config/data_type`. 
They are the final configuration in the composition order, and are what should be specified when running an experiment. 
At a minimum, they should specify the experiment name and model to be run. 

## Examples
### Premise Selection
To run a premise selection experiment, from the root directory of the project simply run:

`python3 -m experiments.premise_selection --config-name=premise_selection/{dataset}/{model}`

where {dataset} is the desired dataset, and {model} is the desired model. 
To change model hyperparameters, modify the appropriate {dataset}/{model} config file. 

### HOList Supervised
To run a premise selection experiment, from the root directory of the project simply run:

`python3 -m experiments.holist_supervised --config-name=holist_supervised/{model}`

### HOList Evaluation
To run a HOList evaluation, from the root directory of the project run:

`python3 -m experiments.holist_eval --config-name=holist_eval/{model}`

There must be a checkpoint file configured which includes the Encoders, Tactic Selection and 
Combiner Networks from the HOList Supervised task. The checkpoint file is specified by the 
`path_model_prefix` field in `configs/experiments/holist_eval/holist_eval.yaml'`, and can be overwritten
from the specific `holist_eval/{model}` file.

The default value, where you can copy HOList supervised checkpoints to is:

`path_model_prefix: 'experiments/holist/checkpoints/checkpoint'`

The first run of the experiment will generate a checkpoint.npy file in the `theorem_embeddings` 
directory specified in the configuration. If the file exists, it will load from the specified directory. 

### TacticZero
To run a TacticZero experiment, from the root directory of the project simply run:

`python3 -m experiments.tacticzero_experiment --config-name=tacticzero/{model}`

## Resuming Runs
To resume a run, you should add the following fields to the final configuration file:

- `exp_config.resume: True`
- `logging_config.id: {wandb_id}` where `wandb_id` is the id associated with the resuming run
- `exp_config.directory: {base_dir}` where `base_dir` is the root of the directory created from the resuming run.
By default, this is in the format: 
    `experiments/runs/${.experiment}/${.name}_${%Y_%m_%d}/${%H_%M_%S}`



# End to End training

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

# Environments 

## holist_env
Wrapper over HOList. Modified to include pretty printed expressions.
Modified proof logging to include pretty printed expressions, replicating original HOList dataset over core and complex
now with PP. 
todo extend this to arbitrary github repos like LeanDojo, and test on flyspeck to reconstruct full HOList benchmark

## leandojo
Wrapper over the standard LeadDojo environment.
Unlike ReProver, this separates subgoals which has several advantages...

