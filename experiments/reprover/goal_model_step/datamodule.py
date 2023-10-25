"""Data module for the tactic generator."""
import pickle
from typing import Optional, List

import pytorch_lightning as pl
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ByT5Tokenizer

from common import (
    Batch,
    Example,
)


class GoalStepDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            max_seq_len: int,
            tokenizer: ByT5Tokenizer,
            critic_tok: str,
            bucket_toks: List[str]
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        self.data = self._load_data(data_path)

        # special tokens for the goal scoring task
        self.critic_tok = critic_tok
        self.bucket_toks = bucket_toks

    def _load_data(self, data_path: str) -> List[Example]:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"{len(data)} examples loaded")


        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]
        return ex

    def collate(self, examples: List[Example]) -> Batch:
        # add the critic token to the beginning of each state to condition the model on the task
        state = [self.critic_tok + ex["goal"] for ex in examples]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        # take targets as the tokens corresponding to the given bucket
        targets = [self.bucket_toks[ex['target']] for ex in examples]

        tokenized_target = self.tokenizer(
            targets,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        target_ids = tokenized_target.input_ids
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        batch = {"state": state,
                 "state_ids": tokenized_state.input_ids,
                 "state_mask": tokenized_state.attention_mask,
                 "targets": targets,
                 "target_ids": target_ids,
                 "target_attention_mask": tokenized_target.attention_mask}

        return batch


class GoalStepDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            num_workers: int,
            critic_tok: str,
            bucket_toks: List[str],
            val_data_path: str = None,
    ) -> None:

        super().__init__()

        self.critic_tok = critic_tok
        self.bucket_toks = bucket_toks

        self.data_path = data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = GoalStepDataset(
                data_path=self.data_path,
                max_seq_len=self.max_seq_len,
                tokenizer=self.tokenizer,
                critic_tok=self.critic_tok,
                bucket_toks=self.bucket_toks,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = GoalStepDataset(
                data_path=self.val_data_path,
                max_seq_len=self.max_seq_len,
                tokenizer=self.tokenizer,
                critic_tok=self.critic_tok,
                bucket_toks=self.bucket_toks,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )