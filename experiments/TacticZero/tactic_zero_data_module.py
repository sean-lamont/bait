import logging
import os
import pickle

import lightning.pytorch as pl
from pymongo import MongoClient
from torch.utils.data import DataLoader as loader
from tqdm import tqdm

from data.HOL4.utils import ast_def
from data.utils.graph_data_utils import transform_expr, transform_batch
from environments.HOL4.tacticzero_old.get_env import get_env


class RLData(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_type = self.config.type
        self.env = get_env(self.config.data_options['environment'])

    def setup(self, stage: str) -> None:
        source = self.config.source

        if source == 'mongodb':
            db = MongoClient()
            db = db[self.config.data_options['db']]
            expr_col = db[self.config.data_options['expression_col']]
            vocab_col = db[self.config.data_options['vocab_col']]
            split_col = db[self.config.data_options['split_col']]

            self.vocab = {v["_id"]: v["index"]
                          for v in tqdm(vocab_col.find({}))
                          }

            self.expr_dict = {v["_id"]: transform_expr({x: v["data"][x] for x in self.config.data_options['filter']},
                                                       data_type=self.data_type,
                                                       vocab=self.vocab,
                                                       config=self.config)
                              for v in tqdm(expr_col.find({}))}

            self.train_goals = [(v['_id'], v['plain']) for v in split_col.find({'split': 'train'})]
            self.test_goals = [(v['_id'], v['plain']) for v in split_col.find({'split': 'val'})]

        elif source == 'directory':
            data_dir = self.config.data_options['directory']
            with open(data_dir, 'rb') as f:
                self.data = pickle.load(f)

            self.vocab = self.data['vocab']
            self.expr_dict = self.data['expr_dict']

            self.expr_dict = {k: transform_expr({x: v[x] for x in self.config.data_options['filter']},
                                                data_type=self.data_type,
                                                vocab=self.vocab,
                                                config=self.config) for k, v in self.expr_dict.items()}

            self.train_goals = self.data[:0.8 * len(self.data)]
            self.test_goals = self.data[0.8 * len(self.data):]

        else:
            raise NotImplementedError

    # only return one goal at a time for TacticZero
    def train_dataloader(self):
        return loader(self.train_goals, batch_size=1, collate_fn=self.setup_goal)

    def val_dataloader(self):
        return loader(self.test_goals, batch_size=1, collate_fn=self.setup_goal)

    # Convert a list of expressions to preprocessed data objects ready for encoding
    def list_to_data(self, data_list):
        if self.data_type == 'fixed':
            return [d.strip().split() for d in data_list]

        # for now, add every unseen expression to database
        for d in data_list:
            if d not in self.expr_dict:
                self.expr_dict[d] = transform_expr(expr=ast_def.goal_to_dict(d),
                                                   data_type=self.data_type,
                                                   vocab=self.vocab,
                                                   config=self.config)

        batch = [self.expr_dict[d] for d in data_list]

        return transform_batch(batch, self.config)

    def gen_fact_pool(self, goal):
        allowed_arguments_ids, candidate_args = self.env.gen_fact_pool(goal)
        allowed_fact_batch = self.list_to_data(candidate_args)

        return allowed_fact_batch, allowed_arguments_ids, candidate_args

    def setup_goal(self, goal):
        goal = goal[0]
        try:
            self.env.reset(goal[1])
        except:
            os.system(command=f'pkill -TERM -P {self.env.process.pid}')
            self.env = get_env(self.config.data_options['environment'])
            return None

        allowed_fact_batch, allowed_arguments_ids, candidate_args = self.gen_fact_pool(goal)

        return goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, self.env

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if batch is None:
            return None
        try:
            goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env = batch
            if self.data_type != 'fixed' and self.data_type != 'sequence':
                allowed_fact_batch = allowed_fact_batch.to(device)
            elif self.data_type == 'sequence':
                allowed_fact_batch = (allowed_fact_batch[0].to(device), allowed_fact_batch[1].to(device))
        except Exception as e:
            logging.debug(f"Error in batch {e}")
            return None
        return goal, allowed_fact_batch, allowed_arguments_ids, candidate_args, env