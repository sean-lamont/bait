import os
import re
import tempfile
from typing import Dict, Any

import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from lightning import pytorch as pl
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from loguru import logger
from transformers import get_cosine_schedule_with_warmup


def get_optimizers(
        parameters, trainer: pl.Trainer, lr: float, warmup_steps: int
) -> Dict[str, Any]:
    """Return an AdamW optimizer with cosine warmup learning rate schedule."""
    strategy = trainer.strategy

    if isinstance(strategy, DeepSpeedStrategy):
        if "offload_optimizer" in strategy.config["zero_optimization"]:
            logger.info("Optimizing with DeepSpeedCPUAdam")
            optimizer = DeepSpeedCPUAdam(parameters, lr=lr, adamw_mode=True)
        else:
            logger.info("Optimizing with FusedAdam")
            optimizer = FusedAdam(parameters, lr=lr, adam_w_mode=True)
    else:
        logger.info("Optimizing with AdamW")
        # optimizer = torch.optim.AdamW(parameters, lr=lr)
        optimizer = torch.optim.AdamW(parameters, lr=lr, eps=1e-4)
        # optimizer = torch.optim.SGD(parameters, lr=lr)

    if trainer.max_steps != -1:
        max_steps = trainer.max_steps
    else:
        assert trainer.max_epochs is not None
        max_steps = (
                trainer.max_epochs
                * len(trainer.datamodule.train_dataloader())
                // trainer.accumulate_grad_batches
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }


def _is_deepspeed_checkpoint(path: str):
    if not os.path.exists(path):
        print(os.stat(path))
        raise FileExistsError(f"Checkpoint {path} does not exist.")
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "zero_to_fp32.py"))


def load_checkpoint(model_cls, ckpt_path: str, device, freeze: bool):
    """Handle DeepSpeed checkpoints in model loading."""
    if not _is_deepspeed_checkpoint(ckpt_path):
        model = model_cls.load_from_checkpoint(ckpt_path, strict=False).to(device)
    else:
        with tempfile.TemporaryDirectory() as dirname:
            path = os.path.join(dirname, "lightning.cpkt")
            convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, path)
            model = model_cls.load_from_checkpoint(path, strict=False)
            model = model.to(device)
    if freeze:
        model.freeze()
    return model


def cpu_checkpointing_enabled(pl_module) -> bool:
    try:
        trainer = pl_module.trainer
        return (
                trainer.strategy is not None
                and isinstance(trainer.strategy, DeepSpeedStrategy)
                and trainer.strategy.config["activation_checkpointing"]["cpu_checkpointing"]
        )
    except RuntimeError:
        return False


Example = Dict[str, Any]
Batch = Dict[str, Any]
MARK_START_SYMBOL = "<a>"
MARK_END_SYMBOL = "</a>"
_SPACES_REGEX = re.compile(r"\s+", re.DOTALL)
