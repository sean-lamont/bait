"""Script for training the goal model."""
import os
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from refactor.goal_model_step.datamodule import GoalStepDataModule
from refactor.goal_model_step.model import StepGoalModel

import ray

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")


def main() -> None:
    ray.init(log_to_driver=False)
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(StepGoalModel, GoalStepDataModule)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
