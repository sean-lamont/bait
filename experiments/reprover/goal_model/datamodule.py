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


class GoalDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            max_seq_len: int,
            tokenizer: ByT5Tokenizer,
            visit_threshold: int,
            critic_tok: str,
            provable_tok: str,
            unprovable_tok: str,

    ) -> None:
        super().__init__()
        self.visit_threshold = visit_threshold
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        self.data = self._load_data(data_path)

        # special tokens for the goal scoring task
        self.critic_tok = critic_tok
        self.provable_tok = provable_tok
        self.unprovable_tok = unprovable_tok

    def _load_data(self, data_path: str) -> List[Example]:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"{len(data)} examples loaded")

        # remove negative samples below the visit_threshold
        neg_data = [d for d in data if d['proved'] == 0 and d['full_visit_count'] > self.visit_threshold]
        pos_data = [d for d in data if d['proved'] == 1]
        data = pos_data + neg_data
        logger.info(f"{len(data)} examples after filtering")

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

        # for now, take targets for negatives as goals which haven't been proven above self.visit_threshold
        targets = [self.provable_tok if ex['proved'] else self.unprovable_tok for ex in examples]

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


class GoalDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            num_workers: int,
            visit_threshold: int,
            critic_tok: str,
            provable_tok: str,
            unprovable_tok: str,
            val_data_path: str = None,
    ) -> None:

        super().__init__()

        self.critic_tok = critic_tok
        self.provable_tok = provable_tok
        self.unprovable_tok = unprovable_tok
        self.visit_threshold = visit_threshold

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
            self.ds_train = GoalDataset(
                data_path=self.data_path,
                max_seq_len=self.max_seq_len,
                tokenizer=self.tokenizer,
                critic_tok=self.critic_tok,
                provable_tok=self.provable_tok,
                unprovable_tok=self.unprovable_tok,
                visit_threshold=self.visit_threshold
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = GoalDataset(
                data_path=self.val_data_path,
                max_seq_len=self.max_seq_len,
                tokenizer=self.tokenizer,
                critic_tok=self.critic_tok,
                provable_tok=self.provable_tok,
                unprovable_tok=self.unprovable_tok,
                visit_threshold=self.visit_threshold
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