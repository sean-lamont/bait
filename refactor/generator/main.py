"""Script for training the tactic generator."""
import os
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from generator.datamodule import GeneratorDataModule
from generator.model import RetrievalAugmentedGenerator


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(RetrievalAugmentedGenerator, GeneratorDataModule)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
