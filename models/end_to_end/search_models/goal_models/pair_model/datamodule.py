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
from experiments.end_to_end.proof_node import ErrorNode, Status, InternalNode
from experiments.end_to_end.stream_dataset import GoalStreamDataset, worker_init_fn


class PairGoalDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            eval_batch_size: int,
            max_seq_len: int,
            critic_tok: str,
            provable_tok: str,
            unprovable_tok: str,
            num_workers=0,
            trace_files=None,
            database='lean_search',
            collection='pair_goal_data',
            replace='keep',  # keep, add or drop to collection if it exists
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

        self.fields = ["positive_goal", "negative_goal"]
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

            # Find proven nodes, label positive and get the most visited sibling that isn't proven as a negative
            # Same for failed nodes, except siblings which are valid as positive

            # todo k negative pairs from siblings rather than 1?
            # todo negative labels for highly explored, unproven nodes, positive for unexplored siblings?
            # todo MLM to sort into semantically similar groups, then label from that

            nodes = trace.nodes
            nodes[trace.tree.goal] = trace.tree

            visits = {node: nodes[node].visit_count for node in nodes.keys()}

            for goal, node in nodes.items():
                for a in node.ancestors:
                    visits[a] += node.visit_count

            # only consider internal nodes with siblings
            proven = [node for node in trace.nodes.values() if node.status == Status.PROVED and node.in_edges]
            failed = [node for node in trace.nodes.values() if node.status == Status.FAILED and node.in_edges]

            def get_most_visited_sibling(node, positive_node, condition):
                dst_nodes = [(d, visits[d.goal]) for x in node.out_edges for d in x.dst if
                             isinstance(d,
                                        InternalNode) and
                             d.goal != positive_node.goal and condition(d) and d.goal in visits]
                if dst_nodes:
                    cur_max = 0
                    cur_node = dst_nodes[0][0]

                    for d in dst_nodes:
                        if cur_max < d[1]:
                            cur_max = d[1]
                            cur_node = d[0]

                    return cur_node
                else:
                    return None

            for node in proven:
                # just take one parent for now
                parent = node.in_edges[0].src
                negative = get_most_visited_sibling(parent, node,
                                                    lambda x: x.status != Status.PROVED)

                if negative:
                    node_data = {'positive_goal': node.data['augmented_state'],
                                 'negative_goal': negative.data['augmented_state'], 'split': split,
                                 'n_visits': visits[negative.goal],
                                 'p_visits': visits[node.goal],
                                 'type': 'proven_pair'}

                    collection.insert_one(node_data)

            for node in failed:
                # just take one parent for now
                parent = node.in_edges[0].src
                positive = get_most_visited_sibling(parent, node,
                                                    lambda x: x.status != Status.FAILED)

                if positive:
                    node_data = {'negative_goal': node.data['augmented_state'],
                                 'n_visits': visits[node.goal],
                                 'positive_goal': positive.data['augmented_state'], 'split': split,
                                 'p_visits': visits[positive.goal],
                                 'type': 'error_pair'}

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
        pos = [self.critic_tok + g['positive_goal'] for g in examples]
        neg = [self.critic_tok + g['negative_gboal'] for g in examples]

        tokenized_pos = self.tokenizer(
            pos,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        tokenized_neg = self.tokenizer(
            neg,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        batch = {"pos_ids": tokenized_pos.input_ids,
                 "pos_mask": tokenized_pos.attention_mask,
                 "target_ids": tokenized_neg.input_ids,
                 "target_attention_mask": tokenized_neg.attention_mask
                 }

        return batch
