"""Data module for the tactic generator."""
import pickle
from pathlib import Path
import random
from typing import Optional

import lightning.pytorch as pl
import torch
from loguru import logger
from pymongo import MongoClient
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from experiments.end_to_end.lightning_common import Batch
from experiments.end_to_end.process_traces import add_rand_idx, filter_traces
from experiments.end_to_end.proof_node import ErrorNode, Status
from experiments.end_to_end.stream_dataset import GoalStreamDataset, worker_init_fn

'''

DataModule for generating tactic embeddings which encode the chances of a tactic being successful

'''


class SeparateTacModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            num_workers: int,
            trace_files=None,
            database='leandojo_novel',
            collection='transitions',
            replace='keep',
            host='localhost:27017'  # mongodb host
    ) -> None:

        super().__init__()

        if trace_files is None:
            trace_files = []
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.fields = ['goal', 'tactic', 'result', 'theorem', 'status', 'time']
        self.collection = collection
        self.database = database
        self.current_train_batch_index = 0
        self.trace_files = trace_files
        self.replace = replace
        self.host = host

    def state_dict(self):
        self.current_train_batch_index = self.ds_train.start_idx
        state = {"current_train_batch_index": self.current_train_batch_index}
        return state

    def load_state_dict(self, state_dict):
        self.current_train_batch_index = state_dict["current_train_batch_index"]
        self.setup()

    def prepare_data(self):
        db = MongoClient(self.host)[self.database]

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

        path = Path(self.trace_files)
        trace_files = [x for x in path.rglob("*") if x.is_file()]

        if not trace_files:
            return

        collection = MongoClient()[self.database][self.collection]

        # split val set at the node level rather than trace level
        def add_trace(trace, split_size):
            nodes = trace.nodes
            nodes[trace.tree.goal] = trace.tree

            random.shuffle(trace.trace)

            for edge in trace.trace[:int(split_size * len(trace.trace))]:
                split = 'train'
                data = {'goal': edge.src.data['augmented_state'], 'tactic': edge.tactic,
                        'logprob': edge.tac_logprob,
                        'split': split,
                        'theorem': trace.theorem.full_name,
                        'time': edge.time}
                if len(edge.dst) == 1 and isinstance(edge.dst[0], ErrorNode):
                    data['result'] = edge.dst[0].inner.message.split(' tactic_state')[0]
                    data['status'] = 'failed'
                else:
                    data['result'] = ''.join([d.goal if hasattr(d, 'goal') else 'Proven' for d in edge.dst])
                    data['status'] = 'success'

                collection.insert_one(data)

            for edge in trace.trace[int(split_size * len(trace.trace)):]:
                split = 'val'
                data = {'goal': edge.src.data['augmented_state'], 'tactic': edge.tactic,
                        'logprob': edge.tac_logprob,
                        'split': split,
                        'theorem': trace.theorem.full_name,
                        'time': edge.time}
                if len(edge.dst) == 1 and isinstance(edge.dst[0], ErrorNode):
                    data['result'] = edge.dst[0].inner.message.split(' tactic_state')[0]
                    data['status'] = 'failed'
                else:
                    data['result'] = ''.join([d.goal if hasattr(d, 'goal') else 'Proven' for d in edge.dst])
                    data['status'] = 'success'

                collection.insert_one(data)

        logger.info('Processing traces for training transition model...')
        for trace in tqdm(trace_files):
            trace = pickle.load(open(trace, 'rb'))
            if isinstance(trace.tree, ErrorNode):
                continue

            add_trace(trace, 0.95)

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
                                              host=self.host
                                              )

        if stage in (None, "fit", "validate"):
            self.ds_val = GoalStreamDataset(db=self.database,
                                            col_name=self.collection,
                                            fields=self.fields,
                                            filter_=val_filter,
                                            gpu_id=self.trainer.global_rank,
                                            num_gpus=self.trainer.num_devices,
                                            host=self.host
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
        # goal = [ex["tactic"] + ex["theorem"] + '\n\n' + ex["goal"][int(len(ex["goal"]) * 0.35):] for ex in examples]
        goal = [ex["theorem"] + '\n\n' + ex["goal"] for ex in examples]
        # goal = [ex["theorem"] + '\n\n' + ex["goal"] for ex in examples]

        tokenized_goal = self.tokenizer(
            goal,
            padding="longest",
            max_length=int(self.max_seq_len * 1.5),
            truncation=True,
            return_tensors="pt",
        )

        result = [ex['status'] + "\n" + ex["result"] for ex in examples]

        tokenized_result = self.tokenizer(
            result,
            padding="longest",
            max_length=min(700, 3000 - tokenized_goal.input_ids.shape[1]),
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

        result_ids = tokenized_result.input_ids
        result_ids[result_ids == self.tokenizer.pad_token_id] = -100

        time_targets = torch.Tensor([ex['time'] for ex in examples])

        batch = {}
        batch["goal"] = goal
        batch["goal_ids"] = tokenized_goal.input_ids
        batch["goal_mask"] = tokenized_goal.attention_mask
        batch["result"] = result
        batch["result_ids"] = tokenized_result.input_ids
        batch["result_mask"] = tokenized_goal.attention_mask
        batch["tactic"] = tactic
        batch["tactic_ids"] = tokenized_tactic.input_ids
        batch["tactic_mask"] = tokenized_tactic.attention_mask
        batch["time_targets"] = time_targets
        batch["status"] = [ex['status'] for ex in examples]

        return batch
