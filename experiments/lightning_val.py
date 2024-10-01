import os
import warnings

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from utils.utils import config_to_dict

warnings.filterwarnings('ignore')

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import torch

"""

Runner for PyTorch Lightning validation

"""


def get_logger(config):
    wandb_logger = WandbLogger(project=config.logging_config.project,
                               name=config.exp_config.name,
                               config=config_to_dict(config),
                               notes=config.logging_config.notes,
                               offline=config.logging_config.offline,
                               save_dir=config.exp_config.directory,
                               )

    return wandb_logger


@hydra.main(config_path="../configs")
def lightning_val(config):
    pl.seed_everything(13231)

    torch.set_float32_matmul_precision('medium')

    OmegaConf.resolve(config)

    os.makedirs(config.exp_config.directory + '/checkpoints', exist_ok=True)

    config = instantiate(config)

    model = config.model

    data_module = config.data_module

    wandb_logger = get_logger(config)

    trainer = pl.Trainer(**config.trainer,
                         logger=wandb_logger,
                         # profiler='advanced'
                         )

    ckpt_dir = config.exp_config.val_ckpt
    trainer.validate(model=model, datamodule=data_module, ckpt_path=ckpt_dir)

    wandb_logger.experiment.finish()
    logger.info(f'Experiment finished')


if __name__ == '__main__':
    lightning_val()
