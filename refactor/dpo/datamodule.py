"""Data module for the tactic generator."""
import copy
import itertools
from itertools import islice
import random
from typing import Optional, List

from loguru import logger
import numpy as np
import time
from pymongo import MongoClient
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import ray
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from refactor.common import (
    Batch,
)
from utils.mongodb_utils import get_batches


class CursorIter(torch.utils.data.IterableDataset):
    def __init__(self, cursor, fields, buf_size=4096):
        super(CursorIter).__init__()
        self.cursor = cursor
        self.batches = get_batches(self.cursor, batch_size=buf_size)
        self.curr_batches = next(self.batches)
        self.remaining = len(self.curr_batches)
        self.fields = fields

    def __iter__(self):
        return self

    def __next__(self):
        if self.remaining == 0:
            self.curr_batches = next(self.batches)
            self.remaining = len(self.curr_batches)
        self.remaining -= 1
        if self.remaining >= 0:
            ret = self.curr_batches.pop()
            return {field: ret[field] for field in self.fields}


# datamodule can pass query which already filters
# todo dpo reloading dataset every 38912??
class GoalStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 db,
                 col_name,
                 fields,
                 range,
                 gpu_id=0,
                 num_gpus=1,
                 worker_id=0,
                 num_workers=1,
                 buf_size=2048,
                 shard_field='rand_idx',
                 start_idx=0):
        super(GoalStreamDataset).__init__()

        self.db = db
        self.col_name = col_name
        self.range = range
        self.fields = fields
        self.buf_size = buf_size
        self.shard_field = shard_field
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus
        self.start_idx = start_idx

        self.query = [{'$match': {self.shard_field: {'$gt': self.range[0], '$lt': self.range[1]}}},
                      {'$sort': {self.shard_field: 1}},
                      {'$project': {v: 1 for v in self.fields}},
                      {'$skip': self.start_idx}]

        if '_id' not in self.fields:
            self.query[-2]['$project']['_id'] = 0

        collection = MongoClient()[self.db][self.col_name]

        # run through once to get the length of cursor
        length = list(collection.aggregate(
            [{'$match': {self.shard_field: {'$gt': self.range[0], '$lt': self.range[1]}}}, {'$count': 'length'}]))[0][
            'length']

        self.length = length // num_gpus

        cursor = collection.aggregate(self.query)

        self.cursor_iter = CursorIter(cursor, fields=self.fields, buf_size=self.buf_size)

        self.setup()

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def reset(self, idx):
        self.__init__(self.db,
                      self.col_name,
                      self.fields,
                      self.range,
                      self.gpu_id,
                      self.num_gpus,
                      self.worker_id,
                      self.num_workers,
                      self.buf_size,
                      self.shard_field,
                      idx)

    def __next__(self):
        try:
            next_ = next(self.ds)
            self.start_idx += 1
            return next_
        except StopIteration:
            self.reset(0)
            return next(self.ds)
        except Exception as e:
            self.reset(self.start_idx)
            logger.warning(f'Loader exception {e}, reloading dataset {len(self)}..')
            return next(self.ds)

    def setup(self):
        total_workers = self.num_gpus * self.num_workers
        global_idx = (self.gpu_id * self.num_workers) + self.worker_id

        # make the dataset iterator return unique values for each worker, and ensure they all have the same number of elements
        self.ds = islice(self.cursor_iter, global_idx, None, total_workers)


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.worker_id = worker_info.id
    dataset.num_workers = worker_info.num_workers
    dataset.setup()


class DPODataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            num_workers: int,
            database='lean_dojo',
            collection='tac_ranks'
    ) -> None:

        super().__init__()

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.fields = ["goal", "winner", "winner_prob", "loser", "loser_prob"]
        self.collection = collection
        self.database = database

    def setup(self, stage: Optional[str] = None) -> None:
        # 90/10 train/val ratio
        train_range = (0., 0.95)
        val_range = (0.95, 1.)

        if stage in (None, "fit"):
            self.ds_train = GoalStreamDataset(db=self.database,
                                              col_name=self.collection,
                                              fields=self.fields,
                                              range=(train_range[0], train_range[1]),
                                              gpu_id=self.trainer.global_rank,
                                              num_gpus=self.trainer.num_devices,
                                              )

        if stage in (None, "fit", "validate"):
            self.ds_val = GoalStreamDataset(db=self.database,
                                            col_name=self.collection,
                                            fields=self.fields,
                                            range=(val_range[0], val_range[1]),
                                            gpu_id=self.trainer.global_rank,
                                            num_gpus=self.trainer.num_devices,
                                            )

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          collate_fn=self.collate_fn,
                          worker_init_fn=worker_init_fn,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size,
                          drop_last=True,
                          pin_memory=True
                          )

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          collate_fn=self.collate_fn,
                          worker_init_fn=worker_init_fn,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size,
                          pin_memory=True
                          )

    def collate_fn(self, examples) -> Batch:
        goals = [g['goal'] for g in examples]
        winners = [g['winner'] for g in examples]
        winner_probs = [g['winner_prob'] for g in examples]
        losers = [g['loser'] for g in examples]
        loser_probs = [g['loser_prob'] for g in examples]

        tokenized_state = self.tokenizer(
            goals,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        tokenized_winners = self.tokenizer(
            winners,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        tokenized_losers = self.tokenizer(
            losers,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        winner_ids = tokenized_winners.input_ids
        winner_ids[winner_ids == self.tokenizer.pad_token_id] = -100

        loser_ids = tokenized_losers.input_ids
        loser_ids[loser_ids == self.tokenizer.pad_token_id] = -100

        batch = {"state": goals,
                 "state_ids": tokenized_state.input_ids,
                 "state_mask": tokenized_state.attention_mask,
                 "winners": winners,
                 "winner_ids": winner_ids,
                 "winner_attention_mask": tokenized_winners.attention_mask,
                 "losers": losers,
                 "loser_ids": loser_ids,
                 "loser_attention_mask": tokenized_losers.attention_mask,
                 "winner_ref_probs": torch.FloatTensor(winner_probs),
                 "loser_ref_probs": torch.FloatTensor(loser_probs)
                 }

        return batch
