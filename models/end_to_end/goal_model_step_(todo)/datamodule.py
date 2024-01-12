"""Data module for the tactic generator."""
import math
from typing import Optional

# import pytorch_lightning as pl
import lightning.pytorch as pl
from loguru import logger
from pymongo import MongoClient
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from experiments.end_to_end.common import (
    Batch,
)
from models.end_to_end.dpo.datamodule import worker_init_fn, GoalStreamDataset
from experiments.end_to_end.process_traces import get_traces
from experiments.end_to_end.proof_node import ErrorNode


# todo test
class GoalLengthDataModule(pl.LightningDataModule):
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
            trace_dir: str,
            database='lean_e2e',
            collection='goal_labels',
            max_len=10,
            visit_threshold=2048
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

        self.fields = ["goal", "target"]
        self.collection = collection
        self.database = database
        self.trace_dir = trace_dir

        self.max_len = max_len
        self.visit_threshold = visit_threshold

    def prepare_data(self):
        traces = get_traces(self.trace_dir)
        logger.info('Processing traces for goal model...')

        for trace in tqdm(traces):
            if isinstance(trace.tree, ErrorNode):
                continue

            nodes = trace.nodes
            nodes[trace.tree.goal] = trace.tree

            visits = {node: nodes[node].visit_count for node in nodes.keys()}

            for goal, node in nodes.items():
                for a in node.ancestors:
                    visits[a] += node.visit_count

            collection = MongoClient()[self.database][self.collection]

            for node in trace.nodes:
                node_data = {'goal': node.goal}
                proof_len = node.distance_to_proof
                if proof_len < self.max_len:
                    node_data['target'] = (self.max_len + 1) - proof_len
                elif proof_len < math.inf:
                    node_data['target'] = 1
                elif visits[node.goal] >= self.visit_threshold:
                    node_data['target'] = 0
                else:
                    continue
                collection.insert_one(node_data)

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
        goals = [g['goal'] for g in examples]
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
