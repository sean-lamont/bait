"""Lightning module for the tactic generator."""
from torchmetrics import Metric
import re
from subprocess import CalledProcessError
from typing import Dict, Any, Optional, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lean_dojo.utils import execute

from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import T5ForConditionalGeneration, AutoTokenizer

from refactor.common import format_augmented_state, zip_strict, remove_marks, get_optimizers, load_checkpoint

torch.set_float32_matmul_precision("medium")


class GenTacModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()

        self.save_hyperparameters()

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

        if config.lora_config:
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
        # pass
        self.run_eval()

    def run_eval(self) -> None:
        ckpt_path = f"{self.trainer.log_dir}/checkpoints/last.ckpt"
        # self.trainer.save_checkpoint(ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

        cmd = f"python -m refactor.new_search_parallel --config-name=leandojo_eval/run num_theorems={self.eval_config.eval_num_theorems}" \
              f" shuffle={self.eval_config.shuffle} timeout={self.eval_config.timeout} tac_model.ckpt_path={ckpt_path}"

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
        self.log("Pass@1_val", acc, on_step=True, on_epoch=True, prog_bar=True)
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
        for i in range(len(state)):
            gen_step = 0
            # keep sampling, flattening distribution until num_samples are generated
            while len(output_text) < num_samples:
                # todo get args from config
                output = self.generator.generate(
                    input_ids=state_ids,
                    attention_mask=state_mask,
                    max_length=self.max_seq_len,
                    do_sample=True,
                    num_return_sequences=num_samples,
                    output_scores=True,
                    return_dict_in_generate=True,
                    top_p=min(1.0, 0.95 + (gen_step * 0.01)),
                    temperature=1.0 + (gen_step * 0.5)
                )

                transitions = self.generator.compute_transition_scores(output.sequences, output.scores,
                                                                       normalize_logits=True)
                # Return the output.
                raw_output_text = self.tokenizer.batch_decode(
                    output.sequences, skip_special_tokens=True
                )

                for j in range(num_samples):
                    t = raw_output_text[j]
                    if t not in output_text:
                        output_text.append(t)
                        score = torch.sum(transitions[j][transitions[j] != -torch.inf]).item()
                        output_score.append(score)

                gen_step += 1

        tactics_with_scores.append(list(zip_strict(output_text, output_score)))

        return tactics_with_scores

    def beamsearch_gen(self, state, state_ids, state_mask, num_samples):
        # Generate tactic candidates using beam search.
        # todo get args from config (length penalty, sample etc.)
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

            # delegate removal of marks to environment
            for j in range(i * num_samples, (i + 1) * num_samples):
                # t = remove_marks(raw_output_text[j])
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


class TopkAccuracy(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, batch_preds: List[List[str]], batch_gt: List[str]):
        assert len(batch_preds) == len(batch_gt)
        for preds, gt in zip(batch_preds, batch_gt):
            # This still doesn't account for short names vs. full names.
            gt = remove_marks(gt)
            preds = [remove_marks(p) for p in preds]
            self.correct += gt in preds[: self.k]
        self.total += len(batch_gt)

    def compute(self) -> float:
        return self.correct.float() / self.total


# todo retriever
class RetrievalAugmentedGenerator(GenTacModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.save_hyperparameters()
        self.num_beams = config.num_beams
        self.length_penalty = config.length_penalty
        ret_ckpt_path = config.ret_ckpt_path

        if ret_ckpt_path is None:
            logger.info("Without retrieval")
            self.retriever = None
        else:
            logger.info(f"Loading the retriever from {ret_ckpt_path}")
            # self.retriever = PremiseRetriever.load(
            #     ret_ckpt_path, self.device, freeze=True
            # )

        self.topk_accuracies = dict()
        for k in range(1, self.num_beams + 1):
            acc = TopkAccuracy(k)
            self.topk_accuracies[k] = acc
            self.add_module(f"top{k}_acc_val", acc)

    def forward(
            self,
            state_ids: torch.Tensor,
            state_mask: torch.Tensor,
            tactic_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=tactic_ids,
        ).loss

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["tactic_ids"],
        )
        self.log(
            "loss_train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
        )

        return loss

    ##############
    # Validation #
    ##############

    def validation_step(self, batch: Dict[str, Any], _) -> None:
        state_ids = batch["state_ids"]
        state_mask = batch["state_mask"]
        tactic_ids = batch["tactic_ids"]

        loss = self(state_ids, state_mask, tactic_ids)
        self.log(f"loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)

        # Generate topk tactic candidates via Beam Search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_seq_len,
            num_beams=self.num_beams,
            do_sample=False,
            num_return_sequences=self.num_beams,
            early_stopping=False,
        )

        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        batch_size = state_ids.size(0)

        assert len(output_text) == batch_size * self.num_beams

        tactics_pred = [
            output_text[i * self.num_beams: (i + 1) * self.num_beams]
            for i in range(batch_size)
        ]

        # Log the topk accuracies.
        for k in range(1, self.num_beams + 1):
            topk_acc = self.topk_accuracies[k]
            topk_acc(tactics_pred, batch["tactic"])
            self.log(f"top{k}_acc_val", topk_acc, on_step=False, on_epoch=True)
