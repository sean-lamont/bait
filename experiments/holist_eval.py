from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import lightning.pytorch as pl

import hydra
import wandb
from omegaconf import OmegaConf

from experiments.holist.process_config import get_prover_options, process_prover_flags
from experiments.holist_pretrain_hydra import config_to_dict

""""

DeepHOL non-distributed prover.

"""

from experiments.holist import prover_runner


@hydra.main(config_path="configs/new_confs", config_name="holist_eval")
def holist_eval(config):
    OmegaConf.resolve(config)

    os.makedirs(config.exp_config.directory + '/checkpoints', exist_ok=True)

    if config.exp_config.resume:
        wandb.init(project=config.logging_config.project,
                   name=config.exp_config.name,
                   config=config_to_dict(config),
                   dir=config.exp_config.directory,
                   resume='must',
                   id=config.logging_config.id,
                   )
    else:
        wandb.init(project=config.logging_config.project,
                   name=config.exp_config.name,
                   config=config_to_dict(config),
                   dir=config.exp_config.directory,
                   )


    prover_options = get_prover_options(config)

    prover_runner.program_started()

    prover_tasks, prover_options, out_path = process_prover_flags(config, prover_options)

    prover_runner.run_pipeline(prover_tasks=prover_tasks,
                               prover_options=prover_options,
                               config=config)


if __name__ == '__main__':
    logging.basicConfig(level=logging.FATAL)
    holist_eval()
