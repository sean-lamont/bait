# Setup Instructions
## Python packages
### Install torch, torch_geometric and PyG supporting packages based on CUDA version (11.7 shown here)
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
- pip install torch_geometric 
- pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
### Install remaining packages from requirements.txt, or from pip as below:
- pip install einops matplotlib plotly igraph pymongo pyrallis wandb dill pyfarmhash absl-py grpcio-tools pexpect torchtext

## Setup Data
You will need a working MongoDB server. To install one locally, you can follow the instructions based on your OS,
with e.g. Ubuntu instructions available here: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/

### Download database dump from ...
We provide preprocessed databases containing data necessary for experiments across all platforms. You can download these at ...
You can select which you want specifically by...

`mongodump --archive=test.20150715.gz --gzip`

Once you have the archive installed, you can run `mongorestore --gzip --archive={archive}` where `archive` is the download.

### Manually process data
The data pipeline is:
- Raw Data from sources. So far we have HOL4, HOLStep, LeanStep, HOList, MIZAR40. You can download these at...
- Run process_{platform} scripts in the data_directory. You will need to specify:
    - The directory of the raw data
    - Whether to use MongoDB or disk (we strongly recommend MongoDB for large datasets such as HOLStep or LeanStep, as it allows for streaming the data without loading it all in memory)
    - What properties to process. We have full sequence for Transformer and sequence based models, and PyG representations for graphs. 
      - Also have other graph related properties such as node depth, and ancestor/descendent nodes for use in Directed SAT models

## Setup

### HOL4
#### Environment
- Download and install polyml: https://polyml.org/download.html 
- We include a prepackaged version of HOL4, which you can build by running environments/hol4/build_hol.sh.
- Alternatively, you can install HOL4 from https://hol-theorem-prover.org/, replacing the hol directory in enviroments/hol4. 
    - (Note that recent versions may not be compatible, the version we used was..)
#### Data
- Raw data included in repository 
- Run process_hol4 from data/hol4 directory

### Lean
Install Lean3:

- Globally
  - wget -q https://raw.githubusercontent.com/leanprover-community/mathlib-tools/master/scripts/install_debian.sh && bash install_debian.sh ; rm -f install_debian.sh && source ~/.profile 
- (preferred) Inside venv:
  - curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
  - source $HOME/.elan/env to add elan to path, or copy/symlink $HOME/.elan/env to venv bin folder
  - pip install mathlibtools
  
#### Environment (Lean-Gym)
- Setup 
#### Data (LeanStep)
- Setup pact
- setup tactic labelled

### HOList
#### Environment
- Setup the Docker container which provides an API to a custom HOL-Light environment setup by Google for AITP.
https://github.com/brain-research/hol-light.git

#### Run HOL light server

- Go to HOL light repo then run:

  - sudo docker build . (returns image_id e.g. 8a85414b942e)
  - sudo docker run -d -p 2000:2000 --name=holist image_id
  - Can restart with sudo docker restart and container_id from above


#### gRPC install/process

- pip install  pip install grpcio-tools
- from root (deepmath-light) dir:
  - python -m grpc_tools.protoc -I=. --python_out=. ./deepmath/deephol/deephol.proto
  - python -m grpc_tools.protoc -I=. --python_out=. ./deepmath/proof_assistant/proof_assistant.proto --grpc_python_out=.


#### Data
- Download raw data from:...

### MIZAR
- Download raw data from ...

### INT
sudo apt-get install libopenmpi-dev
pip install baselines 
pip install git+https://github.com/openai/baselines@ea25b9e8



