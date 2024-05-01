import math
import pickle
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from loguru import logger
from pymongo import MongoClient
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from experiments.end_to_end.common import (
    Batch,
)
from experiments.end_to_end.process_traces import add_rand_idx, filter_traces
from experiments.end_to_end.proof_node import ErrorNode, Status
from experiments.end_to_end.stream_dataset import GoalStreamDataset, worker_init_fn


class HTPSSoftCriticDataModule(pl.LightningDataModule):
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
            trace_files=None,
            database='lean_e2e',
            collection='htps_soft_data',
            replace='keep',  # keep, add or drop if collection exists
    ) -> None:

        super().__init__()

        if trace_files is None:
            trace_files = []

        self.critic_tok = critic_tok
        self.provable_tok = provable_tok
        self.unprovable_tok = unprovable_tok

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.fields = ["goal", "target", "theorem"]
        self.collection = collection
        self.database = database
        self.trace_files = trace_files

        self.current_train_batch_index = 0

        self.replace = replace

    def state_dict(self):
        self.current_train_batch_index = self.ds_train.start_idx
        state = {"current_train_batch_index": self.current_train_batch_index}
        return state

    def load_state_dict(self, state_dict):
        self.current_train_batch_index = state_dict["current_train_batch_index"]
        self.setup()

    def prepare_data(self):
        db = MongoClient()[self.database]

        if self.collection in db.list_collection_names():
            if self.replace == 'keep':
                logger.info('Collection exists, skipping.')
                return
            elif self.replace == 'add':
                logger.info('Collection exists, adding to.')
            elif self.replace == 'drop':
                logger.info('Collection exists, dropping.')
                db[self.collection].drop()
            else:
                raise ValueError(f'Invalid value for replace: {self.replace}')

        logger.info('Loading traces..')

        collection = MongoClient()[self.database][self.collection]

        # trace_files = filter_traces(self.trace_files)

        path = Path(self.trace_files)
        trace_files = [x for x in path.rglob("*") if x.is_file()]

        if not trace_files:
            return

        def add_trace(trace, split):
            nodes = trace.nodes
            nodes[trace.tree.goal] = trace.tree

            edge_data, _, _ = trace.data['search_trace'][-1] if trace.data['search_trace'] else ({}, {}, {})

            edge_nodes = set([edge[0] for edge in edge_data.keys()])

            #  the maximum edge for each goal in the hypergraph
            node_edges = {
                node: max([v['w_score'] / max(1, v['visit_count']) for k, v in edge_data.items() if node == k[0]])
                for node in edge_nodes
            }

            visits = {node: nodes[node].visit_count for node in nodes.keys()}

            for goal, node in nodes.items():
                for a in node.ancestors:
                    visits[a] += node.visit_count

            for node in trace.nodes.values():
                node_data = {'goal': node.goal, 'theorem': trace.theorem.full_name}

                proof_len = node.distance_to_proof

                # score 1 if proven, 0 if failed, otherwise the best edge score from HTPS trace
                if proof_len < math.inf:
                    node_data['target'] = 1
                elif node.status == Status.FAILED:
                    node_data['target'] = 0
                elif node.goal in node_edges and visits[node.goal] > 64:
                    node_data['target'] = node_edges[node.goal]
                else:
                    continue

                node_data['split'] = split
                collection.insert_one(node_data)

        logger.info('Processing traces for training goal model...')
        for file in tqdm(trace_files[:int(0.9 * len(trace_files))]):
            with open(file, 'rb') as f:
                trace = pickle.load(f)

            if isinstance(trace.tree, ErrorNode) or not trace.tree.out_edges:
                continue

            add_trace(trace, 'train')

        logger.info('Processing traces for validating goal model...')

        for file in tqdm(trace_files[int(0.9 * len(trace_files)):]):
            with open(file, 'rb') as f:
                trace = pickle.load(f)

            if isinstance(trace.tree, ErrorNode) or not trace.tree.out_edges:
                continue

            add_trace(trace, 'val')

        add_rand_idx(collection)

    def setup(self, stage: Optional[str] = None) -> None:
        train_filter = [{'$match': {'split': 'train'}},
                        {'$sort': {'rand_idx': 1}}]

        val_filter = [{'$match': {'split': 'val'}},
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
        goals = [g['goal'] for g in examples]
        targets = [g['target'] for g in examples]
        theorems = [g['theorem'] for g in examples]

        # state = [self.critic_tok + ex for ex in goals]

        state = [theorems[i] + self.critic_tok + goals[i] for i in range(len(examples))]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        target_tokens = [self.provable_tok if ex == 1 else self.unprovable_tok for ex in targets]

        tokenized_target = self.tokenizer(
            target_tokens,
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
                 "targets": torch.tensor(targets, dtype=torch.bfloat16),
                 "target_ids": target_ids,
                 "target_attention_mask": tokenized_target.attention_mask
                 }

        return batch
