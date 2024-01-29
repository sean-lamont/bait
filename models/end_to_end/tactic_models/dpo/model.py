"""Lightning module for the tactic generator."""

import re
from subprocess import CalledProcessError
from typing import Dict, Any, Optional, List

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lean_dojo.utils import execute
from loguru import logger
from peft import LoraConfig, get_peft_model
from torchmetrics import Metric
from transformers import T5ForConditionalGeneration, AutoTokenizer

from experiments.end_to_end.common import format_augmented_state, zip_strict, remove_marks, get_optimizers, load_checkpoint

torch.set_float32_matmul_precision("medium")


class GenTacModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.gen_config = config.gen_config
        self.eval_config = config.eval_config
        self.lr = config.lr
        self.warmup_steps = config.warmup_steps
        self.max_seq_len = config.max_seq_len

        # todo more general
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        generator = T5ForConditionalGeneration.from_pretrained(config.model_name)

        # todo
        self.retriever = None

        self.live_eval = config.live_eval if hasattr(config, 'live_eval') else False

        if hasattr(config, 'lora_config') and config.lora_config:
            config = LoraConfig(
                target_modules=list(config.lora_config.target_modules),
                task_type=config.lora_config.task_type,
                r=config.lora_config.r,
                lora_alpha=config.lora_config.lora_alpha,
                lora_dropout=config.lora_config.lora_dropout,
            )
            self.generator = get_peft_model(generator, config)
            logger.info(f"LoRA: ")
            self.generator.print_trainable_parameters()
        else:
            self.generator = generator

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool):
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    def on_fit_start(self) -> None:
        if self.logger is not None and self.global_rank == 0:
            self.logger.log_hyperparams(self.hparams)
            assert self.trainer is not None
            logger.info(f"Logging to {self.trainer.log_dir}")

    ###############################
    # Evaluation with live prover #
    ###############################

    def on_validation_epoch_end(self) -> None:
        if self.global_step > 1 and self.live_eval:
            torch.cuda.empty_cache()
            self.run_eval()

    def run_eval(self) -> None:
        ckpt_path = f"{self.trainer.log_dir}/checkpoints/last_eval.ckpt"
        self.trainer.save_checkpoint(ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

        cmd = f"python -m refactor.new_search_parallel --config-name=leandojo_eval/run num_theorems={self.eval_config.eval_num_theorems}" \
              f" shuffle={self.eval_config.shuffle} env_timeout={self.eval_config.timeout} tac_model.ckpt_path={ckpt_path} log_level='ERROR' tac_model.model='dpo'" \
              f" exp_config.name=eval_{self.global_step} exp_config.experiment=dpo_eval"

        logger.info(f'Running evaluation with {cmd}')

        try:
            # todo better/live output?:
            #  https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
            _, err = execute(cmd, capture_output=True)
        except CalledProcessError as ex:
            logger.error(ex)
            logger.error("Failed to evaluate.")
            return

        m = re.search(r"Pass@1: (\S+)", err)
        assert m is not None, err
        acc = float(m.group(1))
        self.log("Pass@1_val", acc, prog_bar=True)
        logger.info(f"Pass@1: {acc}")

    ##############
    # Prediction #
    ##############

    def generate(self, state: str, retriever_args: dict, num_samples: int):
        return self.batch_generate([state], retriever_args, num_samples)[0]

    def batch_generate(self, state, retriever_args, num_samples):
        if self.retriever is not None:
            retrieved_premises, _ = self.retriever.retrieve(
                state,
                retriever_args['file_path'],
                retriever_args['theorem_full_name'],
                retriever_args['theorem_pos'],
                self.eval_num_retrieved,
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

        if self.gen_config.strategy == 'sample':
            return self.sample_gen(state, state_ids, state_mask, num_samples)
        elif self.gen_config.strategy == 'beam':
            return self.beamsearch_gen(state, state_ids, state_mask, num_samples)

    def sample_gen(self, state, state_ids, state_mask, num_samples):
        # score for nucleus sampling
        tactics_with_scores = []

        output_text = []
        output_score = []
        gen_step = 0

        gen_idx = 0
        # keep sampling until num_samples unique samples are generated, with at most 10 loops
        while len(output_text) < num_samples and gen_idx < 10:
            gen_idx += 1
            output = self.generator.generate(
                input_ids=state_ids,
                attention_mask=state_mask,
                max_length=self.max_seq_len,
                do_sample=True,
                num_return_sequences=num_samples * 2,
                output_scores=True,
                return_dict_in_generate=True,
            )

            transitions = self.generator.compute_transition_scores(output.sequences, output.scores,
                                                                   normalize_logits=True)
            # Return the output.
            raw_output_text = self.tokenizer.batch_decode(
                output.sequences, skip_special_tokens=True
            )

            for j in range(num_samples * 2):
                t = raw_output_text[j]
                if t not in output_text:
                    output_text.append(t)
                    score = torch.sum(transitions[j][transitions[j] != -torch.inf]).item()
                    output_score.append(score)
                if len(output_text) >= num_samples:
                    break

            gen_step += 1

        tactics_with_scores.append(list(zip_strict(output_text, output_score))[:num_samples])

        return tactics_with_scores

    def beamsearch_gen(self, state, state_ids, state_mask, num_samples):
        # Generate tactic candidates using beam search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_seq_len,
            num_beams=num_samples,
            length_penalty=self.gen_config.length_penalty,
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

            for j in range(i * num_samples, (i + 1) * num_samples):
                t = raw_output_text[j]
                if t not in output_text:
                    output_text.append(t)
                    output_score.append(raw_scores[j])

            tactics_with_scores.append(list(zip_strict(output_text, output_score)))

        return tactics_with_scores


class DPOTrainModule(GenTacModel):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.beta = config.beta

        self.save_hyperparameters()

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

        logger.debug(
            f'{torch.cat([winner_pi_probs, batch["winner_ref_probs"], loser_pi_probs, batch["loser_ref_probs"]], dim=0)}')

        self.log(
            "win_rate",
            preferred,
            on_epoch=True,
            sync_dist=True,
            batch_size=winner_pi_probs.shape[0],
            prog_bar=True
        )