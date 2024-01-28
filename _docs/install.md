---
permalink: /docs/install/
title: "Installation"
---

## Requirements

### Install torch, torch_geometric and PyG supporting packages based on CUDA version (11.7 shown here)

```terminal
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

### Run requirements.txt

```terminal pip install -r requirements.txt```

### Install remaining packages

```terminal
pip install pyfarmhash
pip install einops matplotlib plotly igraph pymongo wandb dill pyfarmhash absl-py grpcio-tools pexpect torchtext
```

### MongoDB

You will need a working MongoDB server. To install one locally, you can follow the instructions based on your OS,
with e.g. Ubuntu instructions available here: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/

## Clone repository

Clone the repository. The main branch is the most up to date, and the aaai24 branch contains the code as submitted
for the paper.