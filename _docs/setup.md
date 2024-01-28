---
permalink: /docs/setup/
title: "Setup"
---

# Datasets

## Premise Selection

### [HOL4](/bait/docs/data/#hol4)

- Raw data included in repository
- From the root project directory, run
```terminal
python -m data.hol4.process_hol4
```
### [MIZAR40](/bait/docs/data/#mizar40)
- From the root project directory, run
```terminal 
bash data/mizar/get_mizar.sh
python -m data.mizar.process_mizar
```

### [HOLStep](/bait/docs/data/#holstep)
- From the root project directory, run
```terminal 
bash data/mizar/get_mizar.sh
bash data/holstep/get_holstep.sh
python -m data.holstep.process_holstep
```

### [LeanStep](/bait/docs/data/#leanstep)

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

### [HOList](/bait/docs/data/#holist)

- Obtain data from https://storage.googleapis.com/deepmath/deephol.zip and place it in the `raw_data` directory
  in `data/holist`, then run
```terminal
python -m data.holist.process_holist
```

### [LeanDojo](/bait/docs/data/)

### [INT](/bait/docs/data/)

# Environments

### [HOL4](/bait/docs/enviornments/#hol4)

- Download and install polyml: https://polyml.org/download.html
- Run the HOL4 build script:
```terminal
bash environments/hol4/build_hol.sh
```

### [HOList](/bait/docs/enviornments/#holist)
- Setup the Docker container which provides an API to a custom HOL-Light [environment](https://github.com/brain-research/hol-light.git) setup by Google for AITP .
- Run the setup script: 
```terminal
 bash environments/holist/setup_hollight.sh
```

### [INT](/bait/docs/enviornments/#int)

```terminal
sudo apt-get install libopenmpi-dev
pip install baselines 
pip install git+https://github.com/openai/baselines@ea25b9e8
```
