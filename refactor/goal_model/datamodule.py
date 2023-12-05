"""Data module for the tactic generator."""

from itertools import islice
from typing import Optional

# import pytorch_lightning as pl
import lightning.pytorch as pl
import torch
from loguru import logger
from pymongo import MongoClient
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from refactor.common import (
    Batch,
)
from refactor.dpo.datamodule import worker_init_fn, CursorIter


# todo reloading dataset every 38912??
class GoalStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 db,
                 col_name,
                 fields,
                 filter_,
                 gpu_id=0,
                 num_gpus=1,
                 worker_id=0,
                 num_workers=1,
                 buf_size=2048,
                 start_idx=0):
        super(GoalStreamDataset).__init__()

        self.ds = None
        self.db = db
        self.col_name = col_name
        self.worker_id = worker_id
        self.fields = fields
        self.buf_size = buf_size
        self.filter_ = filter_
        self.num_workers = num_workers
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus
        self.start_idx = start_idx

        self.query = self.filter_ + [{'$project': {v: 1 for v in self.fields}},
                                     {'$skip': self.start_idx}]

        if '_id' not in self.fields:
            self.query[-2]['$project']['_id'] = 0

        collection = MongoClient()[self.db][self.col_name]

        # run through once to get the length of cursor
        length = list(collection.aggregate(
            self.filter_ + [{'$count': 'length'}]))[0][
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
                      self.filter_,
                      self.gpu_id,
                      self.num_gpus,
                      self.worker_id,
                      self.num_workers,
                      self.buf_size,
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

        # make the dataset iterator return unique values for each worker, and ensure they all have the same number of
        # elements
        self.ds = islice(self.cursor_iter, global_idx, None, total_workers)


class GoalProvableDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            num_workers: int,
            critic_tok: str,
            provable_tok: str,
            unprovable_tok: str,
            database='lean_e2e',
            collection='goal_labels'
    ) -> None:

        super().__init__()

        self.critic_tok = critic_tok
        self.provable_tok = provable_tok
        self.unprovable_tok = unprovable_tok

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.fields = ["subgoal", "target"]
        self.collection = collection
        self.database = database

        # self.test_col = MongoClient()['lean_dojo']['test_col']

    def setup(self, stage: Optional[str] = None) -> None:
        # 90/10 train/val ratio
        train_range = (0., 0.9)
        val_range = (0.9, 1.)

        train_filter = [{'$match': {'rand_idx': {'$gt': train_range[0], '$lt': train_range[1]}}},
                        {'$sort': {'rand_idx': 1}}]

        val_filter = [{'$match': {'rand_idx': {'$gt': val_range[0], '$lt': val_range[1]}, 'target': {'$in': [0, 1]}}},
                        {'$sort': {'rand_idx': 1}}]

        if stage in (None, "fit"):
            self.ds_train = GoalStreamDataset(db=self.database,
                                              col_name=self.collection,
                                              fields=self.fields,
                                              filter_=train_filter,
                                              gpu_id=self.trainer.global_rank,
                                              num_gpus=self.trainer.num_devices,
                                              )

        if stage in (None, "fit", "validate"):
            self.ds_val = GoalStreamDataset(db=self.database,
                                            col_name=self.collection,
                                            fields=self.fields,
                                            filter_=val_filter,
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
        # goals = examples['goals']
        # targets = examples['targets']
        # self.test_col.insert_many(examples)
        goals = [g['subgoal'] for g in examples]
        targets = [g['target'] for g in examples]

        state = [self.critic_tok + ex for ex in goals]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        targets = [self.provable_tok if ex == 1 else self.unprovable_tok for ex in targets]

        tokenized_target = self.tokenizer(
            targets,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        # values set to -100 ignored in HuggingFace loss
        target_ids = tokenized_target.input_ids
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        batch = {"state": state,
                 "state_ids": tokenized_state.input_ids,
                 "state_mask": tokenized_state.attention_mask,
                 "targets": targets,
                 "target_ids": target_ids,
                 "target_attention_mask": tokenized_target.attention_mask}

        return batch
