"""Lightning module for the tactic generator."""
from typing import List, Dict, Any

import lightning.pytorch as pl
from torch.nn import CrossEntropyLoss
import torch
from loguru import logger
from transformers import T5ForConditionalGeneration, AutoTokenizer, NoBadWordsLogitsProcessor

from experiments.end_to_end.common import (
    get_optimizers,
    load_checkpoint,
)

torch.set_float32_matmul_precision("medium")


class PairGoalModel(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            lr: float,
            warmup_steps: int,
            max_seq_len: int,
            provable_tok: str,
            unprovable_tok: str,
            critic_tok: str = '<extra_id_0>'
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = T5ForConditionalGeneration.from_pretrained(model_name)

        self.provable_id = self.tokenizer.encode([provable_tok])[0]
        self.unprovable_id = self.tokenizer.encode([unprovable_tok])[0]
        self.critic_tok = critic_tok

        logger.debug(f'provable/unprovable ids: {self.provable_id, self.unprovable_id}')

        # restrict output to just be provable and unprovable
        self.bad_ids = [[i] for i in range(len(self.tokenizer)) if (i != self.provable_id and i != self.unprovable_id)]

        self.logits_processor = NoBadWordsLogitsProcessor(bad_words_ids=self.bad_ids, eos_token_id=None)

    @classmethod
    def load(
            cls, ckpt_path: str, device, freeze: bool
    ) -> "PairGoalModel":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def forward(
            self,
            pos_ids,
            pos_mask,
            neg_ids,
            neg_mask,
            target
    ):
        pos_output = self.generator(
            input_ids=pos_ids,
            attention_mask=pos_mask,
            labels=target)

        filtered_logits = self.logits_processor(pos_ids, pos_output.logits)

        pos_provable_logit = filtered_logits[:, 0, self.provable_id]
        pos_unprovable_logit = filtered_logits[:, 0, self.unprovable_id]

        neg_output = self.generator(
            input_ids=neg_ids,
            attention_mask=neg_mask,
            labels=target)

        filtered_logits = self.logits_processor(neg_ids, neg_output.logits)

        neg_provable_logit = filtered_logits[:, 0, self.provable_id]
        neg_unprovable_logit = filtered_logits[:, 0, self.unprovable_id]

        return pos_provable_logit, neg_provable_logit, pos_unprovable_logit, neg_unprovable_logit

    def pair_loss(self,
                  pos_provable_logit,
                  neg_provable_logit,
                  pos_unprovable_logit,
                  neg_unprovable_logit) -> torch.Tensor:
        loss = (torch.log(1 + torch.exp(-1 * (pos_provable_logit - neg_provable_logit)))
                + torch.log(1 + torch.exp(-1 * (neg_unprovable_logit - pos_unprovable_logit))))

        return torch.sum(loss)

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        pos_provable, neg_provable, pos_unprovable, neg_unprovable = self(
            batch["pos_ids"],
            batch["pos_mask"],
            batch["neg_ids"],
            batch["neg_mask"],
            batch["target"])

        loss = self.pair_loss(pos_provable, neg_provable, pos_unprovable, neg_unprovable)

        self.log(
            "loss_train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True,
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

    def validation_step(self, batch, batch_idx: int):
        pos_provable, neg_provable, pos_unprovable, neg_unprovable = self(
            batch["pos_ids"],
            batch["pos_mask"],
            batch["neg_ids"],
            batch["neg_mask"],
            batch["target"])

        loss = self.pair_loss(pos_provable, neg_provable, pos_unprovable, neg_unprovable)

        # frequency of times positive is ranked higher than negative

        pos_acc = torch.sum(pos_provable > neg_provable) / len(pos_provable)
        neg_acc = torch.sum(neg_unprovable > pos_unprovable) / len(pos_provable)

        self.log(
            "pos_acc",
            pos_acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        self.log(
            "neg_acc",
            neg_acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        self.log(
            "loss_val",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        return loss

    ##############
    # Prediction #
    ##############

    def generate(self, state: str) -> float:
        return self.batch_generate([state])[0]

    def batch_generate(self, state: List[str]) -> List[float]:
        # concat the token indicating this is a goal task
        state = [self.critic_tok + s for s in state]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        output = self.generator.generate(
            input_ids=state_ids,
            max_new_tokens=1,
            bad_words_ids=self.bad_ids,
            attention_mask=state_mask,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        filtered_logits = self.logits_processor(state_ids, output.scores[0])

        probs = torch.log_softmax(filtered_logits, dim=1)

        # get the logits for the provable index
        provable_prob = probs[:, self.provable_id]

        return provable_prob
