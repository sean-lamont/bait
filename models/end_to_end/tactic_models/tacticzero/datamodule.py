"""Data module for the tactic generator."""
import math
import pickle
from typing import Optional

import lightning.pytorch as pl
from loguru import logger
from pymongo import MongoClient
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from experiments.end_to_end.common import (
    Batch,
)
from experiments.end_to_end.process_traces import add_rand_idx, filter_traces
from experiments.end_to_end.proof_node import ErrorNode
from experiments.end_to_end.stream_dataset import GoalStreamDataset, worker_init_fn


class TacticZeroDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            num_workers: int,
            trace_files=None,
            database='lean_e2e',
            collection='seq2seq'
    ) -> None:

        super().__init__()

        if trace_files is None:
            trace_files = []
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.fields = ['goal', 'tactic']
        self.collection = collection
        self.database = database
        self.current_train_batch_index = 0
        self.trace_files = trace_files

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
            logger.info('Collection exists, dropping.')
            db[self.collection].drop()

        logger.info('Loading traces..')

        trace_files = filter_traces(self.trace_files)
        traces = []

        for file in tqdm(trace_files):
            with open(file, 'rb') as f:
                traces.append(pickle.load(f))

        if not traces:
            return

        collection = MongoClient()[self.database][self.collection]

        # todo process nodes to extract arguments and tactics (goal will be handled in search_model)
        # just string processing on edge tactics, then process reward similarly to ILQL
        def add_trace(trace, split):

            nodes = trace.nodes
            nodes[trace.tree.goal] = trace.tree

            for node in nodes.values():
                if node.out_edges:
                    # get best edge
                    distance = min(edge.distance_to_proof() for edge in node.out_edges)
                    # edge is proving
                    if distance < math.inf:
                        # take first element to break ties
                        edge = [e for e in node.out_edges if e.distance_to_proof() == distance][0]
                        collection.insert_one({'goal': node.goal, 'tactic': edge.tactic, 'split': split})

        logger.info('Processing traces for training TacticZero model...')
        for trace in tqdm(traces[:int(0.9 * len(traces))]):
            if isinstance(trace.tree, ErrorNode):
                continue

            add_trace(trace, 'train')

        logger.info('Processing traces for validating TacticZero model...')
        for trace in tqdm(traces[int(0.9 * len(traces)):]):
            if isinstance(trace.tree, ErrorNode):
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
        state = [ex["goal"] for ex in examples]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        tactic = [ex["tactic"] for ex in examples]

        tokenized_tactic = self.tokenizer(
            tactic,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        tactic_ids = tokenized_tactic.input_ids
        tactic_ids[tactic_ids == self.tokenizer.pad_token_id] = -100

        batch = {}
        batch["state"] = state
        batch["state_ids"] = tokenized_state.input_ids
        batch["state_mask"] = tokenized_state.attention_mask
        batch["tactic"] = tactic
        batch["tactic_ids"] = tactic_ids
        batch["tactic_mask"] = tokenized_tactic.attention_mask

        # Copy other fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch
