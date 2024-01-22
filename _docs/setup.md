---
permalink: /docs/setup/
title: "Setup"
---

# Premise Selection 
## HOL4
- Raw data included in repository
- Run `python -m data.hol4.process_hol4` from the root project directory
 
## MIZAR
- Run `bash data/mizar/get_mizar.sh`
- Run `python -m data.mizar.process_mizar`
 
## HOLStep
- Run `bash data/holstep/get_holstep.sh`
- Run `python -m data.holstep.process_holstep`

## LeanStep
Install Lean3:

- Globally
  - `wget -q https://raw.githubusercontent.com/leanprover-community/mathlib-tools/master/scripts/install_debian.sh && bash install_debian.sh ; rm -f install_debian.sh && source ~/.profile`
- (preferred) Inside venv:
  - `curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh`
  - `source $HOME/.elan/env` to add elan to path, or copy/symlink `$HOME/.elan/env` to venv bin folder
  - `pip install mathlibtools`

- Run `bash data/lean/lean-step/setup.sh` from project root (this is a large dataset, you can
  terminate the parallel_data_gen script early for a reduced dataset)
- Run `python -m data.lean.process_leanstep`

# HOL4
- Download and install polyml: https://polyml.org/download.html
- Run the HOL4 build script `bash environments/hol4/build_hol.sh`

# HOList
## Environment
- Setup the Docker container which provides an API to a custom HOL-Light environment setup by Google for AITP.
- Run `bash environments/holist/setup_hollight.sh`
- (Obtained from https://github.com/brain-research/hol-light.git, with licence in directory)

## Data
- obtain data from https://storage.googleapis.com/deepmath/deephol.zip and place it in a `raw_data` directory in `data/holist`
- Run `python -m data.holist.process_holist`

# LeanDojo
## Data
Run script.


<!-- 
[//]: # (### INT)

[//]: # (sudo apt-get install libopenmpi-dev)

[//]: # (pip install baselines )

[//]: # (pip install git+https://github.com/openai/baselines@ea25b9e8)

[//]: # () -->
