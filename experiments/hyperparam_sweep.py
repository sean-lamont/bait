import os
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint

from data.get_data import get_data
from experiments.premise_selection_wandb import premise_selection_experiment

warnings.filterwarnings('ignore')
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from models.get_model import get_model
from models.gnn.formula_net.formula_net import BinaryClassifier
import torch
from experiments.pyrallis_configs_old import PremiseSelectionConfig

# @hydra.main(config_path="./", config_name="sweep")
def sweep(config):
    config = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    sweep_id = wandb.sweep(sweep=config, project="test_sweep")
    wandb.agent(sweep_id=sweep_id, function=premise_selection_experiment)

if __name__ == '__main__':
    sweep()
