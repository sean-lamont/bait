"""Data module for the tactic generator."""
import math
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


class ILQLDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            num_workers: int = 0,
            database='leandojo_initial',
            collection='edge_data'
    ) -> None:

        super().__init__()

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.fields = ["goal", "tactic", "distance_to_proof", "top_proven", "outcome"]
        self.collection = collection
        self.database = database

    def setup(self, stage: Optional[str] = None) -> None:
        train_range = (0., 0.95)
        val_range = (0.95, 1.)

        train_filter = [{'$match': {'rand_idx': {'$gt': train_range[0], '$lt': train_range[1]}}},
                        {'$sort': {'rand_idx': 1}}]

        val_filter = [{'$match': {'rand_idx': {'$gt': val_range[0], '$lt': val_range[1]}}},
                      {'$sort': {'rand_idx': 1}}]

        if stage in (None, "fit"):
            self.ds_train = GoalStreamDataset(db=self.database,
                                              col_name=self.collection,
                                              fields=self.fields,
                                              filter_=train_filter,
                                              # gpu_id=self.trainer.global_rank,
                                              # num_gpus=self.trainer.num_devices,
                                              )

        if stage in (None, "fit", "validate"):
            self.ds_val = GoalStreamDataset(db=self.database,
                                            col_name=self.collection,
                                            fields=self.fields,
                                            filter_=val_filter,
                                            # gpu_id=self.trainer.global_rank,
                                            # num_gpus=self.trainer.num_devices,
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

    def reward_func(self, example):
        if example['outcome'][0] == 'Error':
            return -0.1
        elif example['distance_to_proof'] < math.inf:
            if example['top_proven']:
                # original goal proven
                return 10
            else:
                # only subgoal proven
                return 1
        # if tactic application successful
        else:
            return 0.1

    def collate_fn(self, examples) -> Batch:
        state = [g['goal'] for g in examples]
        tactics = [g['tactic'] for g in examples]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        tokenized_target = self.tokenizer(
            tactics,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        target_ids = tokenized_target.input_ids
        # values set to -100 ignored in HuggingFace loss
        # target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        sequence_rewards = [self.reward_func(g) for g in examples]

        # rewards will be batch_size x seq_len
        rewards = torch.zeros(target_ids.shape[0], max(target_ids.shape[1] - 1, 1))

        # sum of attention_mask gives # of tokens in sequence, and reward ignores the first tok
        tok_counts = torch.sum(tokenized_target.attention_mask, dim=1) - 2

        # set the reward at the index of the last token
        for i in range(rewards.shape[0]):
            if tok_counts[i] >= 0:
                rewards[i, tok_counts[i]] = sequence_rewards[i]
            else:
                rewards[i, 0] = sequence_rewards[i]


        batch = {"tokens": tokenized_state.input_ids,
                 "attn_mask": tokenized_state.attention_mask,
                 "target": target_ids,
                 "target_mask": tokenized_target.attention_mask,
                 "rewards": rewards}

        return batch


# if __name__ == '__main__':
#     module = ILQLDataModule(model_name='kaiyuy/leandojo-lean3-tacgen-byt5-small',
#                             batch_size=2,
#                             eval_batch_size=2,
#                             max_seq_len=2048
#                             )
#
#     module.setup()
#
#     batches = []
#     for i, batch in enumerate(module.train_dataloader()):
#         batches.append(batch)
#         if i == 10:
#             break
