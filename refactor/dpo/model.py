"""Lightning module for the tactic generator."""

from typing import List, Dict, Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import T5ForConditionalGeneration, AutoTokenizer

from experiments.reprover.common import (
    get_optimizers,
    load_checkpoint,
)
from refactor.common import zip_strict, format_augmented_state

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

    ##############
    # Prediction #
    ##############

    def generate(
            self,
            state: str,
            retriever_args: dict,
            num_samples: int,
    ) -> List[Tuple[str, float]]:
        return self.batch_generate(
            [state], retriever_args, num_samples
        )[0]

    def batch_generate(
            self,
            state: List[str],
            retriever_args: dict,
            num_samples: int,
    ) -> List[List[Tuple[str, float]]]:

        logger.debug(state)

        if self.retriever is not None:
            retrieved_premises, _ = self.retriever.retrieve(
                state=state,
                **retriever_args,
                # self.eval_num_retrieved,
            )
            state = [
                format_augmented_state(s, premises, self.max_seq_len, p_drop=0.0)
                for s, premises in zip_strict(state, retrieved_premises)
            ]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        # Generate tactic candidates using beam search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_seq_len,
            num_beams=num_samples,
            length_penalty=self.length_penalty,
            do_sample=False,
            num_return_sequences=num_samples,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        raw_scores = output.sequences_scores.tolist()
        tactics_with_scores = []

        for i in range(len(state)):
            output_text = []
            output_score = []

            # delegate removal of marks to environment
            for j in range(i * num_samples, (i + 1) * num_samples):
                # t = remove_marks(raw_output_text[j])
                t = raw_output_text[j]
                if t not in output_text:
                    output_text.append(t)
                    output_score.append(raw_scores[j])

            tactics_with_scores.append(list(zip_strict(output_text, output_score)))

        return tactics_with_scores
