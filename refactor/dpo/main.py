"""Script for training the goal model."""
import os

from refactor.dpo.datamodule import DPODataModule
from refactor.dpo.model import DPOTrainModule
from loguru import logger
from pytorch_lightning.cli import LightningCLI


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(DPOTrainModule, DPODataModule)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
