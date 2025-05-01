"""Lightning module for the tactic generator."""

import re
from subprocess import CalledProcessError
from typing import Dict, Any
from typing import Optional, List

import torch
from lean_dojo.utils import execute
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger
from torchmetrics import Metric
from torchmetrics.text import SacreBLEUScore

from experiments.end_to_end.common import remove_marks, zip_strict, format_augmented_state
from models.end_to_end.tactic_models.gen_tac_model import GenTacModel

torch.set_float32_matmul_precision("medium")


# subclass of ModelCheckpoint, which saves the checkpoint using HF save_pretrained
# https://github.com/Lightning-AI/pytorch-lightning/issues/19228
class HFModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath):
        logger.info("Saving checkpoint to %s", filepath)
        trainer.lightning_module.generator.save_pretrained(filepath)
        # super()._save_checkpoint(trainer, filepath)


class SFTModel(GenTacModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.save_hyperparameters()

    def on_train_start(self):
        self.generator.hf_device_map = {'': self.device}

    # standard causal LM loss
    def forward(
            self,
            state_ids: torch.Tensor,
            state_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=state_ids,
        ).loss

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        # if self.global_rank == 0 and self.global_step % 50000 == 0:
        #     ckpt_path = f"{self.trainer.log_dir}/checkpoints/last_eval.ckpt"
        #     self.trainer.lightning_module.generator.save_pretrained(ckpt_path)

        loss = self(
            batch["state_ids"],
            batch["state_mask"],
        )
        self.log(
            "loss_train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        return loss

    ##############
    # Validation #
    ##############

    def validation_step(self, batch: Dict[str, Any], _) -> None:
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
        )
        self.log(
            "loss_val",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        return loss


