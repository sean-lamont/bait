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


def get_logger(config):
    if config.exp_config.resume:
        logger.info('Resuming run..')
        wandb_logger = WandbLogger(project=config.logging_config.project,
                                   name=config.exp_config.name,
                                   config=config_to_dict(config),
                                   notes=config.logging_config.notes,
                                   offline=config.logging_config.offline,
                                   save_dir=config.exp_config.directory,
                                   id=config.logging_config.id,
                                   resume='must',
                                   )

    else:
        wandb_logger = WandbLogger(project=config.logging_config.project,
                                   name=config.exp_config.name,
                                   config=config_to_dict(config),
                                   notes=config.logging_config.notes,
                                   offline=config.logging_config.offline,
                                   save_dir=config.exp_config.directory,
                                   )

    return wandb_logger


@hydra.main(config_path="../configs")
def supervised_runner(config):
    torch.set_float32_matmul_precision('medium')

    OmegaConf.resolve(config)

    os.makedirs(config.exp_config.directory + '/checkpoints', exist_ok=True)

    config = instantiate(config)

    model = config.model

    data_module = config.data_module

    wandb_logger = get_logger(config)

    trainer = pl.Trainer(**config.trainer,
                         logger=wandb_logger)

    if config.exp_config.resume:
        ckpt_dir = config.exp_config.checkpoint_dir + "/last.ckpt"
        trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_dir)
    else:
        trainer.fit(model=model, datamodule=data_module)

    wandb_logger.experiment.finish()
    logger.info(f'Experiment finished')

    # logs the saved checkpoint with $ delimiter to allow for a parent process to find it.
    # todo take best validation checkpoint instead
    logger.error(f'checkpoint_dir: {config.exp_config.checkpoint_dir}/last.ckpt' + '$')

    # todo add option for testing/validating the trainer


if __name__ == '__main__':
    supervised_runner()
