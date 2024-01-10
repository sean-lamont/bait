"""Data module for the tactic generator."""
import math
from typing import Optional

# import pytorch_lightning as pl
import lightning.pytorch as pl
import torch
from loguru import logger
from pymongo import MongoClient
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from refactor.common import (
    Batch,
)
from refactor.goal_model.datamodule import GoalStreamDataset
from refactor.process_traces import get_traces, add_rand_idx
from refactor.proof_node import ErrorNode, ProofFinishedNode, Status
from refactor.stream_dataset import worker_init_fn


class ILQLDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            num_workers: int = 0,
            trace_dir='',
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

        self.trace_dir = trace_dir

    def prepare_data(self):
        traces = get_traces(self.trace_dir)

        if not traces:
            return

        logger.info('Processing traces for goal model...')

        collection = MongoClient()[self.database][self.collection]

        for trace in tqdm(traces):
            if isinstance(trace.tree, ErrorNode):
                continue

            for i, edge in enumerate(trace.trace):
                datum = {
                    'goal': edge.src.goal,
                    'tactic': edge.tactic,
                    'distance_to_proof': edge.distance_to_proof(),
                    'top_proven': trace.tree.status == Status.PROVED}

                if len(edge.dst) == 1 and isinstance(edge.dst[0], ErrorNode):
                    datum['outcome'] = ['Error']
                elif len(edge.dst) == 1 and isinstance(edge.dst[0], ProofFinishedNode):
                    datum['outcome'] = ['Proven']
                else:
                    outcome = [d.goal for d in edge.dst]
                    datum['outcome'] = outcome
                collection.insert_one(datum)

        add_rand_idx(collection)

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
