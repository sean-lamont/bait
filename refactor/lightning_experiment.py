import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

warnings.filterwarnings('ignore')

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import torch

# todo option to suppress output from imports

def config_to_dict(conf):
    return OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )


def get_logger(config):
    if config.exp_config.resume:
        logging.info('Resuming run..')
        logger = WandbLogger(project=config.logging_config.project,
                             name=config.exp_config.name,
                             config=config_to_dict(config),
                             notes=config.logging_config.notes,
                             offline=config.logging_config.offline,
                             save_dir=config.exp_config.directory,
                             id=config.logging_config.id,
                             resume='must',
                             )

    else:
        logger = WandbLogger(project=config.logging_config.project,
                             name=config.exp_config.name,
                             config=config_to_dict(config),
                             notes=config.logging_config.notes,
                             offline=config.logging_config.offline,
                             save_dir=config.exp_config.directory,
                             )

    return logger


@hydra.main(config_path="../experiments/configs/experiments")
def lightning_experiment(config):
    torch.set_float32_matmul_precision('medium')

    OmegaConf.resolve(config)

    os.makedirs(config.exp_config.directory + '/checkpoints', exist_ok=True)

    config = instantiate(config)

    experiment = config.experiment

    data_module = config.data_module

    logger = get_logger(config)

    trainer = pl.Trainer(**config.trainer,
                         logger=logger)

    if config.exp_config.resume:
        ckpt_dir = config.exp_config.checkpoint_dir + "/last.ckpt"
        trainer.fit(model=experiment, datamodule=data_module, ckpt_path=ckpt_dir)
    else:
        trainer.fit(model=experiment, datamodule=data_module)

    logger.experiment.finish()

    # todo convert model to standard checkpoint/save to HF


if __name__ == '__main__':
    lightning_experiment()
