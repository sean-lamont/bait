"""Data module for the tactic generator."""
from typing import Optional, List

from loguru import logger
import numpy as np
import pytorch_lightning as pl
import torch
import ray
from transformers import AutoTokenizer

from refactor.common import (
    Batch,
)


class GoalStepDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            num_workers: int,
            critic_tok: str,
            bucket_toks: List[str],
            database='lean_dojo',
            collection='goal_len_task'
    ) -> None:

        super().__init__()

        self.critic_tok = critic_tok
        self.bucket_toks = bucket_toks

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
            ds = ray.data.read_mongo(
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
            self.ds_train = ds

        if stage in (None, "fit", "validate"):
            ds = ray.data.read_mongo(
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
            self.ds_val = ds

    def train_dataloader(self):
        return self.ds_train.iter_torch_batches(collate_fn=self.collate_fn, batch_size=self.batch_size)

    def val_dataloader(self):
        return self.ds_val.iter_torch_batches(collate_fn=self.collate_fn, batch_size=self.eval_batch_size)

    def collate_fn(self, examples) -> Batch:
        goals = np.copy(examples['goal'])
        targets = torch.from_numpy(np.copy(examples['target']))


        # add the critic token to the beginning of each state to condition the model on the task
        # state = [self.critic_tok + ex["goal"] for ex in goals]
        state = [self.critic_tok + ex for ex in goals]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        # take targets as the tokens corresponding to the given bucket
        # targets = [self.bucket_toks[ex['target']] for ex in examples]
        targets = [self.bucket_toks[ex] for ex in targets]

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

# if __name__ == '__main__':
#     module = GoalStepDataModule(model_name='kaiyuy/leandojo-lean3-tacgen-byt5-small',
#                                 batch_size=2,
#                                 eval_batch_size=4,
#                                 max_seq_len=2300,
#                                 num_workers=2,
#                                 critic_tok='<extra_id_0>',
#                                 bucket_toks=['<extra_id_1>', '<extra_id_2>', '<extra_id_3>', '<extra_id_4>',
#                                              '<extra_id_5>', '<extra_id_6>',
#                                              '<extra_id_7>', '<extra_id_8>', '<extra_id_9>',
#                                              '<extra_id_10>', '<extra_id_11>'])
#
#     module.setup()
#
#     train_loader = module.train_dataloader()
#
#
