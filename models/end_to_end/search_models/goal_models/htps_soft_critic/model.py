"""Lightning module for the tactic generator."""
from typing import List, Dict, Any

import lightning.pytorch as pl
from torch.nn import CrossEntropyLoss
import torch
from loguru import logger
from torchmetrics.classification import BinaryConfusionMatrix
from transformers import T5ForConditionalGeneration, AutoTokenizer, NoBadWordsLogitsProcessor

from experiments.end_to_end.lightning_common import get_optimizers, load_checkpoint

torch.set_float32_matmul_precision("medium")


class HTPSSoftCritic(pl.LightningModule):
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

        self.ce_loss = CrossEntropyLoss()

        self.bcm = BinaryConfusionMatrix()  # normalize='true')

    @classmethod
    def load(
            cls, ckpt_path: str, device, freeze: bool
    ) -> "HTPSSoftCritic":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def forward(
            self,
            state_ids: torch.Tensor,
            state_mask: torch.Tensor,
            target_ids: torch.Tensor,
            soft_targets: torch.Tensor
    ) -> torch.Tensor:
        output = self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=target_ids)

        filtered_logits = self.logits_processor(state_ids, output.logits)

        provable_logit = filtered_logits[:, 0, self.provable_id]
        unprovable_logit = filtered_logits[:, 0, self.unprovable_id]

        logits = torch.stack([provable_logit, unprovable_logit], dim=1)

        soft_targets = torch.stack([soft_targets, 1 - soft_targets], dim=1)

        loss = self.ce_loss(logits, soft_targets)

        return loss

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["target_ids"],
            batch["targets"]
        )

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
        output = self.generator(
            input_ids=batch['state_ids'],
            attention_mask=batch['state_mask'],
            labels=batch['target_ids'])

        filtered_logits = self.logits_processor(batch['state_ids'], output.logits)

        provable_logit = filtered_logits[:, 0, self.provable_id]
        unprovable_logit = filtered_logits[:, 0, self.unprovable_id]

        logits = torch.stack([unprovable_logit, provable_logit], dim=1)

        inds, preds = torch.max(logits, dim=1)

        # consider accuracy wrt rounded labels
        targets = (batch['targets'] >= 0.5).long().to(self.device)
        confusion = self.bcm(preds, targets.long())

        acc = torch.sum((preds == targets) / (preds == targets).shape[0])

        self.log(
            "val_acc",
            acc,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        self.log(
            "false_negs",
            confusion[1][0],
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        self.log(
            "true_negs",
            confusion[0][0],
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        self.log(
            "false_pos",
            confusion[0][1],
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        self.log(
            "true_pos",
            confusion[1][1],
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        return

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
