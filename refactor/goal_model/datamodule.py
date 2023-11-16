"""Data module for the tactic generator."""
import itertools
from itertools import islice
from random import random
from typing import Optional, List

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
            database='lean_dojo',
            collection='goal_proven_task'
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

        self.goal_fields = {"_id": 0, "goal": 1, "target": 1}
        self.collection = collection
        self.database = database

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            ds_train = ray.data.read_mongo(
                uri='localhost:27017',
                database=self.database,
                collection=self.collection,
                # sort by random index so data is shuffled, and constant between cursors
                pipeline=[{'$match': {'rand_idx': {'$lt': 0.9}}},
                          {"$sort": {'rand_idx': 1}},
                          {'$project': self.goal_fields},
                          ],

            )

            # logger.warning(ds.count())
            # self.ds_train = ds.split(self.trainer.world_size, equal=True)[self.trainer.global_rank]
            self.ds_train = ds_train

        if stage in (None, "fit", "validate"):
            ds_val = ray.data.read_mongo(
                uri='localhost:27017',
                database=self.database,
                collection=self.collection,
                # sort by random index so data is shuffled, and constant between cursors
                pipeline=[{'$match': {'rand_idx': {'$gt': 0.9}}},
                          {"$sort": {'rand_idx': 1}},
                          {'$project': self.goal_fields},
                          ],

            )

            # self.ds_val = ds.split(self.trainer.world_size, equal=True)[self.trainer.global_rank]
            self.ds_val = ds_val

    def train_dataloader(self):
        return self.ds_train.iter_torch_batches(collate_fn=self.collate_fn, batch_size=self.batch_size)

    def val_dataloader(self):
        return self.ds_val.iter_torch_batches(collate_fn=self.collate_fn, batch_size=self.eval_batch_size)

    def collate_fn(self, examples) -> Batch:
        goals = np.copy(examples['goal'])
        targets = np.copy(examples['target'])

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

        target_ids = tokenized_target.input_ids
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        batch = {"state": state,
                 "state_ids": tokenized_state.input_ids,
                 "state_mask": tokenized_state.attention_mask,
                 "targets": targets,
                 "target_ids": target_ids,
                 "target_attention_mask": tokenized_target.attention_mask}

        return batch


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
            # random.shuffle(self.curr_batches)
            self.remaining = len(self.curr_batches)
        self.remaining -= 1
        if self.remaining >= 0:
            ret = self.curr_batches.pop()
            return {field: ret[field] for field in self.fields}


# datamodule can pass query which already filters
class GoalStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self, db, col_name, fields, range, buf_size=4096, shard_field='rand_idx'):
        super(GoalStreamDataset).__init__()

        self.db = db
        self.col_name = col_name
        self.range = range
        self.fields = fields
        self.buf_size = buf_size
        self.shard_field = shard_field
        self.setup()

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.ds)

    def setup(self):
        self.collection = MongoClient()[self.db][self.col_name]

        # run through once to get the length of cursor
        self.query = [{'$match': {self.shard_field: {'$gt': self.range[0], '$lt': self.range[1]}}},
                      {'$sort': {self.shard_field: 1}},
                      {'$project': {v: 1 for v in self.fields}}]

        if '_id' not in self.fields:
            self.query[-1]['$project']['_id'] = 0

        self.length = list(self.collection.aggregate(
            [{'$match': {self.shard_field: {'$gt': self.range[0], '$lt': self.range[1]}}}, {'$count': 'length'}]))[0][
            'length']

        self.cursor = self.collection.aggregate(self.query)

        self.ds = itertools.cycle(CursorIter(self.cursor, fields=self.fields, buf_size=self.buf_size))


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    num_workers = worker_info.num_workers

    start, stop = dataset.range
    size = stop - start

    stop = start + (((worker_id + 1) / num_workers) * size)
    start = start + ((worker_id / num_workers) * size)

    dataset.range = (start, stop)
    dataset.setup()


# todo test with datamodule
# prevent Lightning to inject DDSampler
# Trainer(replace_sampler_ddp=False)


# split dataset from trainer world size in module, then pass ray dataset as input for iterable dataset?
if __name__ == '__main__':
    # module = GoalProvableDataModule(model_name='kaiyuy/leandojo-lean3-tacgen-byt5-small',
    #                                 batch_size=2,
    #                                 eval_batch_size=4,
    #                                 num_workers=2,
    #                                 max_seq_len=2300,
    #                                 critic_tok='<extra_id_0>',
    #                                 provable_tok='<extra_id_1>',
    #                                 unprovable_tok='<extra_id_2>',
    #                                 )
    #
    # module.setup()
    # ray_ds = module.ds_train
    #
    # # mongo_iter = GoalStreamDataset(data=ray_ds.streaming_split(1))
    # mongo_iter = GoalStreamDataset(data=ray_ds)
    #
    # # loader1 = DataLoader(mongo_iter, batch_size=2, num_workers=0, worker_init_fn=worker_init_fn)
    # loader1 = DataLoader(mongo_iter, batch_size=2, pin_memory=True)  # , num_workers=0, worker_init_fn=worker_init_fn)
    #
    # data_ = []
    # for a in tqdm(iter(loader1)):
    #     data_.append(a)

    db = 'hol4'
    col = 'split_data'
    ds = GoalStreamDataset(db=db,
                           col_name=col,
                           buf_size=2048,
                           fields=['split', 'stmt'],
                           range=(0, 0.5),
                           )

    loader = DataLoader(ds, worker_init_fn=worker_init_fn, num_workers=8, batch_size=2)
    t0 = time.monotonic()

    data = []
    for i in tqdm(loader):
        data.extend(i)

    print('total time: {t0}, size: {len(data)}')
# class GoalStreamDataset(torch.utils.data.IterableDataset):
#     def __init__(self, data):
#         super().__init__()
#         self.data = data
#
#     def __len__(self):
#         return self.data.count()
#
#     def __iter__(self):
#         return self.data.iter_rows()
#

# class GoalStreamModule(pl.LightningDataModule):
#     def __init__(
#             self,
#             model_name: str,
#             batch_size: int,
#             eval_batch_size: int,
#             max_seq_len: int,
#             num_workers: int,
#             critic_tok: str,
#             provable_tok: str,
#             unprovable_tok: str,
#             database='lean_dojo',
#             collection='goal_proven_task'
#     ) -> None:
#         super().__init__()
#
#         self.critic_tok = critic_tok
#         self.provable_tok = provable_tok
#         self.unprovable_tok = unprovable_tok
#
#         self.batch_size = batch_size
#         self.eval_batch_size = eval_batch_size
#         self.max_seq_len = max_seq_len
#         self.num_workers = num_workers
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#         self.goal_fields = {"_id": 0, "goal": 1, "target": 1}
#         self.collection = collection
#         self.database = database
#
#         self.ds_train = GoalStreamDataset(ray.data.read_mongo(
#             uri='localhost:27017',
#             database=self.database,
#             collection=self.collection,
#             # sort by random index so data is shuffled, and constant between cursors
#             pipeline=[{'$match': {'rand_idx': {'$lt': 0.9}}},
#                       {"$sort": {'rand_idx': 1}},
#                       {'$project': self.goal_fields},
#                       ],
#
#         ))
#
#         self.ds_val = GoalStreamDataset(ray.data.read_mongo(
#             uri='localhost:27017',
#             database=self.database,
#             collection=self.collection,
#             # sort by random index so data is shuffled, and constant between cursors
#             pipeline=[{'$match': {'rand_idx': {'$gt': 0.9}}},
#                       {"$sort": {'rand_idx': 1}},
#                       {'$project': self.goal_fields},
#                       ],
#
#         ))
#
#     def train_dataloader(self):
#         return DataLoader(
#             self.ds_train,
#             self.batch_size,
#             # num_workers=self.num_workers,
#             num_workers=0,
#             shuffle=False,
#             pin_memory=True,
#             drop_last=True,
#             collate_fn=self.collate_fn
#         )
#
#     def val_dataloader(self):
#         return DataLoader(
#             self.ds_val,
#             self.eval_batch_size,
#             # num_workers=self.num_workers,
#             num_workers=0,
#             shuffle=False,
#             pin_memory=True,
#             drop_last=True,
#             collate_fn=self.collate_fn
#         )
#
#     def collate_fn(self, examples) -> Batch:
#         # goals = np.copy(examples['goal'])
#         # targets = np.copy(examples['target'])
#
#         goals = [g['goal'] for g in examples]
#         targets = [g['target'] for g in examples]
#
#         state = [self.critic_tok + ex for ex in goals]
#
#         tokenized_state = self.tokenizer(
#             state,
#             padding="longest",
#             max_length=self.max_seq_len,
#             truncation=True,
#             return_tensors="pt",
#         )
#
#         targets = [self.provable_tok if ex == 1 else self.unprovable_tok for ex in targets]
#
#         tokenized_target = self.tokenizer(
#             targets,
#             padding="longest",
#             max_length=self.max_seq_len,
#             truncation=True,
#             return_tensors="pt",
#         )
#
#         target_ids = tokenized_target.input_ids
#         target_ids[target_ids == self.tokenizer.pad_token_id] = -100
#
#         batch = {"state": state,
#                  "state_ids": tokenized_state.input_ids,
#                  "state_mask": tokenized_state.attention_mask,
#                  "targets": targets,
#                  "target_ids": target_ids,
#                  "target_attention_mask": tokenized_target.attention_mask}
#
#         return batch
#
