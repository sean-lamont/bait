from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import hydra
import wandb
from omegaconf import OmegaConf

from experiments.holist.utils.process_config import get_prover_options, process_prover_flags
from experiments.holist_supervised import config_to_dict
import experiments.holist.simple_prover_runner as prover_runner

""""

DeepHOL non-distributed prover.

"""


@hydra.main(config_path="configs/experiments")#, config_name="experiments/holist_eval")
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
                   mode='offline' if config.logging_config.offline else 'online'
                   )
    else:
        wandb.init(project=config.logging_config.project,
                   name=config.exp_config.name,
                   config=config_to_dict(config),
                   dir=config.exp_config.directory,
                   mode='offline' if config.logging_config.offline else 'online'
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
