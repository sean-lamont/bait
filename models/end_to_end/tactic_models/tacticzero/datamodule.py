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

'''

Data module for training TacticZero on traces produced in End-to-End training.

---- Currently not implemented for any specific training regime. 
Can be either policy gradient based as done originally, 
or could be modified to use e.g. supervised learning on successful proofs. ---- 

'''


class TacticZeroDataModule(pl.LightningDataModule):
    def __init__(
            self
    ) -> None:

        super().__init__()

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

        # todo process nodes here to extract arguments and tactics (goal will be handled in search_model)
        def add_trace(trace, split):
            pass

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

    # todo
    def collate_fn(self, examples) -> Batch:
        state = [ex["goal"] for ex in examples]

        tactic = [ex["tactic"] for ex in examples]

        return
