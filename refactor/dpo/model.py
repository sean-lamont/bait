"""Lightning module for the tactic generator."""
from subprocess import CalledProcessError
from typing import Dict, Any

import time
import re
import pytorch_lightning as pl
from lean_dojo.utils import execute
import torch
import torch.nn.functional as F
from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import T5ForConditionalGeneration, AutoTokenizer

from experiments.reprover.common import (
    get_optimizers,
    load_checkpoint,
)

torch.set_float32_matmul_precision("medium")


# todo better abstract tactic generation class, with eval, generate, batch_generate
# todo run evaluation (similar to generator eval)
class DPOTrainModule(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            lr: float,
            warmup_steps: int,
            max_seq_len: int,
            beta: float,
            num_beams=64,
            eval_num_retrieved=100,
            eval_num_cpus=1,
            eval_num_theorems=200,
            length_penalty: float = 0.0,
            ret_ckpt_path=None,

    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.num_beams = num_beams
        self.eval_num_retrieved = eval_num_retrieved
        self.eval_num_cpus = eval_num_cpus
        self.eval_num_theorems = eval_num_theorems
        self.length_penalty = length_penalty
        self.ret_ckpt_path = ret_ckpt_path
        self.lr = lr
        self.beta = beta
        self.warmup_steps = warmup_steps
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator = T5ForConditionalGeneration.from_pretrained(model_name)

        target_modules = ['q', 'k', 'v', 'o', 'wo', 'lm_head']
        config = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            r=16,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.01,
        )
        self.generator = get_peft_model(generator, config)
        logger.info(f"LoRA: ")
        self.generator.print_trainable_parameters()
        self.retriever = None

    @classmethod
    def load(
            cls, ckpt_path: str, device, freeze: bool
    ) -> "DPOTrainModule":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def dpo_loss(self, pi_yw_logps, pi_yl_logps, ref_yw_logps, ref_yl_logps):
        """
        pi_yw_logps: policy logprobs for winners
        ref_yw_logps: reference model logprobs for winners
        pi_yl_logps: policy logprobs for losers
        ref_yl_logps: reference model logprobs for losers
        beta: temperature controlling strength of KL penalty
        """

        # pi_logps = torch.cat([pi_yw_logps, pi_yl_logps], dim=0)
        # ref_logps = torch.cat([ref_yw_logps, ref_yl_logps], dim=0)
        # rewards = beta * (pi_logps - ref_logps).detach()

        pi_logratios = pi_yw_logps - pi_yl_logps
        ref_logratios = ref_yw_logps - ref_yl_logps
        losses = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios))
        return losses  # , rewards

    def forward(
            self,
            state_ids: torch.Tensor,
            state_mask: torch.Tensor,
            winner_ids: torch.Tensor,
            loser_ids: torch.Tensor,
            winner_ref_probs: torch.Tensor,
            loser_ref_probs: torch.Tensor,
    ) -> torch.Tensor:
        winner_pi_logits = self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=winner_ids,
        ).logits

        loser_pi_logits = self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=loser_ids
        ).logits

        winner_ids[winner_ids == -100] = 0
        loser_ids[loser_ids == -100] = 0

        # construct probability for whole sequence (from testing, slightly off reported beam search scores by ~10^-2)
        normalised = torch.log_softmax(winner_pi_logits, -1)
        outs = torch.gather(normalised, -1, winner_ids.unsqueeze(-1)).squeeze(-1)
        outs[winner_ids == self.tokenizer.pad_token_id] = 0
        winner_pi_probs = torch.sum(outs, dim=1)

        normalised = torch.log_softmax(loser_pi_logits, -1)
        outs = torch.gather(normalised, -1, loser_ids.unsqueeze(-1)).squeeze(-1)
        outs[loser_ids == self.tokenizer.pad_token_id] = 0
        loser_pi_probs = torch.sum(outs, dim=1)

        loss = self.dpo_loss(winner_pi_probs, loser_pi_probs, winner_ref_probs, loser_ref_probs)

        return torch.sum(loss, dim=0)

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["winner_ids"],
            batch["loser_ids"],
            batch["winner_ref_probs"],
            batch["loser_ref_probs"],

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

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    def on_fit_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            assert self.trainer is not None
            logger.info(f"Logging to {self.trainer.log_dir}")

    ##############
    # Validation #
    ##############

    def validation_step(self, batch, batch_idx):
        state_ids = batch['state_ids']
        state_mask = batch['state_mask']
        winner_ids = batch['winner_ids']
        loser_ids = batch['loser_ids']

        winner_pi_logits = self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=winner_ids,
        ).logits

        loser_pi_logits = self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=loser_ids
        ).logits

        winner_ids[winner_ids == -100] = 0
        loser_ids[loser_ids == -100] = 0

        # construct probability for whole sequence (from testing, slightly off reported beam search scores by ~10^-2)
        normalised = torch.log_softmax(winner_pi_logits, -1)
        outs = torch.gather(normalised, -1, winner_ids.unsqueeze(-1)).squeeze(-1)
        outs[winner_ids == self.tokenizer.pad_token_id] = 0
        winner_pi_probs = torch.sum(outs, dim=1)

        normalised = torch.log_softmax(loser_pi_logits, -1)
        outs = torch.gather(normalised, -1, loser_ids.unsqueeze(-1)).squeeze(-1)
        outs[loser_ids == self.tokenizer.pad_token_id] = 0
        loser_pi_probs = torch.sum(outs, dim=1)

        # when difference is > 0, winner_probs is greater
        preferred = sum((winner_pi_probs - loser_pi_probs) > 0) / winner_pi_probs.shape[0]

        logger.info(
            f'{torch.cat([winner_pi_probs, batch["winner_ref_probs"], loser_pi_probs, batch["loser_ref_probs"]], dim=0)}')

        self.log(
            "win_rate",
            preferred,
            on_epoch=True,
            sync_dist=True,
            batch_size=winner_pi_probs.shape[0],
            prog_bar=True
        )

    #########################
    # End to End Validation #
    #########################

    def on_validation_epoch_end(self) -> None:
        ckpt_path = f"{self.trainer.log_dir}/checkpoints/last.ckpt"
        self.trainer.save_checkpoint(ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

        cmd = f"python -m refactor.new_search_parallel --config-name=leandojo_eval/run num_theorems={self.eval_num_theorems}" \
              f" shuffle=true timeout=60"

        logger.info(f'Running evaluation with {cmd}')

        wait_time = 3600
        while True:
            try:
                # todo better/live output: https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-runningk
                _, err = execute(cmd, capture_output=True)
                break
            except CalledProcessError as ex:
                logger.error(ex)
                logger.error(
                    f"Failed to evaluate. Retrying in {wait_time / 3600} hour..."
                )
                time.sleep(wait_time)
                wait_time *= 2

        m = re.search(r"Pass@1: (\S+)", err)
        assert m is not None, err
        acc = float(m.group(1))
        self.log("Pass@1_val", acc, on_step=False, on_epoch=True)
        logger.info(f"Pass@1: {acc}")

    # todo parameterise generation (e.g shared function which takes generation config, and outputs tactic with scores)
