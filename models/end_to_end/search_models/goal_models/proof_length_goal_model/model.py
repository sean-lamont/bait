"""Lightning module for goal model."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

import einops
import pytorch_lightning as pl
import torch
from loguru import logger
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, AutoTokenizer, NoBadWordsLogitsProcessor

from experiments.reprover.common import (
    get_optimizers,
    load_checkpoint,
)

torch.set_float32_matmul_precision("medium")

mseloss = torch.nn.MSELoss()


class GoalModel(ABC):
    """A tactic generator takes a state and generates multiple tactic candidates."""

    @abstractmethod
    def generate(
            self,
            state: str,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def batch_generate(
            self,
            state: List[str],
    ) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError


# model which predicts the number of (bucketed) steps needed to prove a goal
class StepGoalModel(GoalModel, pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            lr: float,
            warmup_steps: int,
            max_seq_len: int,
            bucket_toks: List[str]

    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = T5ForConditionalGeneration.from_pretrained(model_name)

        logger.debug(f'Bucket Tokens: {bucket_toks}')

        # ignore last token as EOS
        self.bucket_ids = self.tokenizer.encode(bucket_toks)[:-1]

        logger.debug(f'Bucket IDs: {self.bucket_ids}')

        # restrict output to be only step buckets
        self.bad_ids = [[i] for i in range(len(self.tokenizer)) if i not in self.bucket_ids]
        self.logits_processor = NoBadWordsLogitsProcessor(bad_words_ids=self.bad_ids, eos_token_id=None)

        weights = torch.zeros(384)

        # todo take as parameter
        weights[260:271] = torch.Tensor(
            [0.39694545454545455,
             90.21487603305785,
             58.37433155080214,
             39.694545454545455,
             38.16783216783217,
             24.2039911308204,
             9.729055258467023,
             4.594276094276094,
             2.0252319109461965,
             0.8606796499250966,
             0.15664777211738537]
        )

        self.ce_loss = CrossEntropyLoss(ignore_index=1, weight=weights.bfloat16().to(self.device))

        self.mseloss = torch.nn.MSELoss()

    @classmethod
    def load(
            cls, ckpt_path: str, device, freeze: bool
    ) -> "StepGoalModel":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def forward(
            self,
            state_ids: torch.Tensor,
            state_mask: torch.Tensor,
            target_ids: torch.Tensor,
    ) -> torch.Tensor:

        output = self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=target_ids)

        filtered_logits = self.logits_processor(state_ids, output.logits)


        # get probs for the buckets
        filtered_logits = filtered_logits[:, 0, self.bucket_ids]
        buckets = torch.Tensor(self.bucket_ids).bfloat16().to(self.device)
        probs = torch.softmax(filtered_logits, dim=1)

        # get the weighted sum over buckets as the final score
        batch_size = probs.shape[0]

        buckets = einops.repeat(buckets, 'd -> n d', n=batch_size).bfloat16()

        score = torch.sum(probs * (buckets - 260),
                          dim=1) #/ torch.LongTensor(len(self.bucket_ids) - 1).to(self.device)

        targ_score = (target_ids[:, 0] - 260).bfloat16() #/ (len(self.bucket_ids) - 1)

        loss = self.mseloss(score, targ_score)

        # print (f'score {score}, actual {targ_score}')
        #
        # print (f'score {score} targets {(targets - 260) / (len(self.bucket_ids) - 1)}')

        return loss

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["target_ids"],
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
        self._log_io_texts("train", batch["state_ids"], batch["target_ids"])
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    def _log_io_texts(
            self,
            split: str,
            state_ids: torch.LongTensor,
            target_ids: torch.LongTensor,
    ) -> None:
        tb = self.logger.experiment

        inp = self.tokenizer.decode(state_ids[0], skip_special_tokens=False)

        oup_ids = torch.where(
            target_ids[0] == -100, self.tokenizer.pad_token_id, target_ids[0]
        )

        oup = self.tokenizer.decode(oup_ids, skip_special_tokens=False)

        tb.add_text(f"{split}_state", f"```\n{inp}\n```", self.global_step)

        tb.add_text(f"{split}_tactic", f"`{oup}`", self.global_step)

    def on_fit_start(self) -> None:
        ckpt_path = f"{self.trainer.log_dir}/checkpoints/last.ckpt"
        self.trainer.save_checkpoint(ckpt_path)

        logger.info(f"Saved checkpoint to {ckpt_path}")

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

        # ignore the EOS at the end
        assert batch['target_ids'].shape[1] == filtered_logits.shape[1] == 2

        filtered_logits = filtered_logits[:, 0]

        # Take the highest predicted bucket as the prediction

        preds = torch.max(filtered_logits, dim=1)[1]
        targets = batch['target_ids'][:, 0]

        print (f'{preds}, {targets}')

        acc = torch.sum((preds == targets) / (preds == targets).shape[0])

        output = self.generator.generate(
            input_ids=batch['state_ids'],
            max_new_tokens=1,
            bad_words_ids=self.bad_ids,
            attention_mask=batch['state_mask'],
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # shape = (batch, tokens)
        filtered_logits = self.logits_processor(batch['state_ids'], output.scores[0])

        # get probs for the buckets
        filtered_logits = filtered_logits[:, self.bucket_ids]
        buckets = torch.LongTensor(self.bucket_ids).to(self.device)
        probs = torch.softmax(filtered_logits, dim=1)

        # get the weighted sum over buckets as the final score
        batch_size = probs.shape[0]
        buckets = einops.repeat(buckets, 'd -> n d', n=batch_size)
        score = torch.sum(probs * (buckets - 260), dim=1) / (len(self.bucket_ids) - 1)

        # print(f'score {score}, preds {preds}, actual {targets}')

        print(f'score {score} targets {(targets - 260) / (len(self.bucket_ids) - 1)}')

        self.log(
            "val_score",
            mseloss(score, (targets - 260) / (len(self.bucket_ids) - 1)),
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

        self.log(
            "val_acc",
            acc,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
            prog_bar=True
        )

    ##############
    # Prediction #
    ##############

    def generate(self, state: str) -> float:
        return self.batch_generate([state])[0]

    def batch_generate(self, state: List[str]) -> Tensor:
        logger.debug(state)

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

        # get probs for the buckets
        filtered_logits = filtered_logits[:, self.bucket_ids]
        buckets = torch.LongTensor(self.bucket_ids).to(self.device)
        probs = torch.softmax(filtered_logits, dim=1)

        # get the weighted sum over buckets as the final score
        batch_size = probs.shape[0]
        buckets = einops.repeat(buckets, 'd -> n d', n=batch_size)
        score = torch.sum(probs * (buckets - 260), dim=1) / (len(self.bucket_ids) - 1)

        return torch.log(score)
