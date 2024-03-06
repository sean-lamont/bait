import math
import pickle
import random
from typing import Optional

import torch
from lightning.pytorch import LightningDataModule
from loguru import logger
from pymongo import MongoClient
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.HOList.utils import io_util
from data.HOList.utils import theorem_fingerprint as fp
from data.HOList.utils.sexpression_to_graph import sexpression_to_graph
from data.utils.graph_data_utils import transform_batch
from data.utils.graph_data_utils import transform_expr
from experiments.end_to_end.process_traces import filter_traces, add_rand_idx
from experiments.end_to_end.proof_node import ErrorNode
from experiments.end_to_end.stream_dataset import GoalStreamDataset
from experiments.end_to_end.stream_dataset import worker_init_fn

'''

HOList DataModule adapted for End-to-End pipeline. 

To train the model on initial human proofs, use the DataModule from the original HOList experiment,
found in models/HOList/supervised/datamodule.py, then use the checkpoint for End-to-End.

'''


class HOListDataModule(LightningDataModule):
    def __init__(self,
                 config):
        super().__init__()
        self.config = config
        self.batch_size = self.config.batch_size
        self.current_train_batch_index = 0

        self.database = self.config.data_options['db']

        self.vocab_col = config.data_options['vocab_col']
        self.collection = config.data_options['collection']
        self.thms_col = config.data_options['thms_col']

        self.trace_files = config.trace_files if config.trace_files else []

    def state_dict(self):
        self.current_train_batch_index = self.ds_train.start_idx
        state = {"current_train_batch_index": self.current_train_batch_index}
        return state

    def load_state_dict(self, state_dict):
        self.current_train_batch_index = state_dict["current_train_batch_index"]
        self.setup()

    # process traces to be in correct format for HOList model training
    def prepare_data(self):
        db = MongoClient()[self.database]

        if self.collection in db.list_collection_names():
            logger.info('Collection exists, dropping.')
            db[self.collection].drop()

        logger.info('Loading traces..')

        if not self.trace_files:
            return

        traces = []

        trace_files = filter_traces(self.trace_files)

        for file in tqdm(trace_files):
            with open(file, 'rb') as f:
                traces.append(pickle.load(f))

        if not traces:
            return

        collection = MongoClient()[self.database][self.collection]

        tactics = io_util.load_tactics_from_file(str(self.config.tactics_path), self.config.path_tactics_replace)

        tactics_name_id_map = {tactic.name: tactic.id for tactic in tactics}

        theorem_db = io_util.load_theorem_database_from_file(self.config.theorem_dir)

        fingerprint_conclusion_map = {
            fp.Fingerprint(theorem): theorem.conclusion
            for theorem in theorem_db.theorems}

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

                        # strip brackets of tactic
                        tactic_str = edge.tactic.replace('[', '')
                        tactic_str = tactic_str.replace(']', '')

                        tactic = tactic_str.split(' ')[0]
                        args = tactic_str.split(' THM')[1:]

                        logger.info(edge.tactic, tactic, args)

                        args = [fingerprint_conclusion_map[int(param.split(' ')[1])] for param in args]

                        collection.insert_one({'goal': node.goal,
                                               'tac_id': tactics_name_id_map[tactic],
                                               'thms': args,
                                               'split': split,
                                               'thms_hard_negatives': [],})

        logger.info('Processing traces for training seq2seq model...')
        for trace in tqdm(traces[:int(0.9 * len(traces))]):
            if isinstance(trace.tree, ErrorNode):
                continue

            add_trace(trace, 'train')

        logger.info('Processing traces for validating seq2seq model...')
        for trace in tqdm(traces[int(0.9 * len(traces)):]):
            if isinstance(trace.tree, ErrorNode):
                continue

            add_trace(trace, 'val')

        add_rand_idx(collection)

    def to_data(self, expr):
        return transform_expr(expr, self.config.type, self.vocab, self.config)

    def list_to_data(self, data_list):
        # process expression to desired format, only computing expensive depth and attention_edge if specified in conf
        data_dict = [sexpression_to_graph(x, self.config.type, 'attention_edge' in self.config.data_options[
            'filter'] or 'depth' in self.config.data_options.filter) for x in data_list]

        batch = [self.to_data({x: v[x] for x in self.config.data_options['filter']}) for v in data_dict]

        return transform_batch(batch, config=self.config)

    def collate_fn(self, examples):
        # examples will be a list of proof step dictionaries with goal, thms, tactic_id
        goals = [x['goal'] for x in examples]

        goals = self.list_to_data(goals)

        pos_thms = [random.choice(x['thms']) if len(x['thms']) > 0
                    else 'NO_PARAM'
                    for x in examples]

        pos_thms = self.list_to_data(pos_thms)

        tacs = torch.LongTensor([x['tac_id'] for x in examples])

        # random negative samples per goal
        neg_thms = [[a for a in random.sample(self.thms_ls, self.batch_size - 1)] for _ in range(self.batch_size)]

        neg_thms = [self.list_to_data(th) for th in neg_thms]

        return goals, tacs, pos_thms, neg_thms

    def setup(self, stage: Optional[str] = None) -> None:
        db = MongoClient()[self.config.data_options['db']]
        db_name = self.config.data_options['db']
        vocab_col = db[self.config.data_options['vocab_col']]
        split_col = self.config.data_options['collection']
        thms_col = db[self.config.data_options['thms_col']]

        self.vocab = {v["_id"]: v["index"]
                      for v in tqdm(vocab_col.find({}))
                      }

        self.thms_ls = [v['_id'] for v in thms_col.find({})]

        fields = ['goal', 'thms', 'tac_id', 'thms_hard_negatives']

        train_filter = [{'$match': {'split': 'train'}},
                        {'$sort': {'rand_idx': 1}}]

        val_filter = [{'$match': {'split': 'val'}},
                      {'$sort': {'rand_idx': 1}}]

        if stage in (None, "fit"):
            self.ds_train = GoalStreamDataset(db=db_name,
                                              col_name=split_col,
                                              fields=fields,
                                              filter_=train_filter,
                                              gpu_id=self.trainer.global_rank,
                                              num_gpus=self.trainer.num_devices,
                                              )

        if stage in (None, "fit", "validate"):
            self.ds_train = GoalStreamDataset(db=db_name,
                                              col_name=split_col,
                                              fields=fields,
                                              filter_=train_filter,
                                              gpu_id=self.trainer.global_rank,
                                              num_gpus=self.trainer.num_devices,
                                              )

            self.ds_val = GoalStreamDataset(db=db_name,
                                            col_name=split_col,
                                            fields=fields,
                                            filter_=val_filter,
                                            gpu_id=self.trainer.global_rank,
                                            num_gpus=self.trainer.num_devices,
                                            )

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          collate_fn=self.collate_fn,
                          worker_init_fn=worker_init_fn,
                          num_workers=self.config.num_workers,
                          batch_size=self.batch_size,
                          drop_last=True,
                          pin_memory=True
                          )

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          collate_fn=self.collate_fn,
                          worker_init_fn=worker_init_fn,
                          num_workers=self.config.num_workers,
                          batch_size=self.batch_size,
                          pin_memory=True
                          )
