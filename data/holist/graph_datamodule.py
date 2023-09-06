import pickle
import random

import torch
from lightning.pytorch import LightningDataModule
from pymongo import MongoClient
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.stream_dataset import MongoStreamDataset
from data.utils.graph_data_utils import list_to_data
from data.utils.graph_data_utils import to_data


class HOListDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = self.config.batch_size

    def setup(self, stage: str) -> None:
        source = self.config.source
        if source == 'mongodb':
            db = MongoClient()
            db = db[self.config.data_options['db']]
            expr_col = db[self.config.data_options['expression_col']]
            vocab_col = db[self.config.data_options['vocab_col']]
            split_col = db[self.config.data_options['split_col']]
            thms_col = db[self.config.data_options['thms_col']]

            self.vocab = {v["_id"]: v["index"]
                          for v in tqdm(vocab_col.find({}))
                          }

            self.thms_ls = [v['_id'] for v in thms_col.find({})]

            fields = ['goal', 'thms', 'tac_id', 'thms_hard_negatives']

            # load all examples in memory, otherwise keep as a cursor
            if self.config.data_options['dict_in_memory']:
                self.expr_dict = {v["_id"]: self.to_data({x: v["data"][x] for x in self.config.data_options['filter']})
                                  for v in tqdm(expr_col.find({}))}
            else:
                self.expr_col = expr_col

            # if data_in_memory, save all examples to disk
            if self.config.data_options['split_in_memory']:
                self.train_data = [{field: v[field] for field in fields}
                                   for v in tqdm(split_col.find({'split': 'train', 'source': 'human'}))]

                self.val_data = [{field: v[field] for field in fields}
                                 for v in tqdm(split_col.find({'split': 'val', 'source': 'human'}))]

            # stream dataset from MongoDB
            else:
                train_len = split_col.count_documents({'split': 'train', 'source': 'human'})
                val_len = split_col.count_documents({'split': 'val', 'source': 'human'})

                self.train_data = MongoStreamDataset(split_col.find({'split': 'train', 'source': 'human'}),
                                                     len=train_len,
                                                     fields=fields)

                self.val_data = MongoStreamDataset(split_col.find({'split': 'val', 'source': 'human'}), len=val_len,
                                                   fields=fields)


        elif source == 'directory':
            data_dir = self.config.data_options['directory']
            with open(data_dir, 'rb') as f:
                self.data = pickle.load(f)

            self.vocab = self.data['vocab']
            self.expr_dict = self.data['expr_dict']
            self.expr_dict = {k: self.to_data(v) for k, v in self.expr_dict.items()}

            self.train_data = self.data['train_data']
            self.val_data = self.data['val_data']
            self.thms_ls = self.data['train_thm_ls']

        else:
            raise NotImplementedError


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.gen_batch)#, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.gen_batch)#, num_workers=1)

    def to_data(self, expr):
        return to_data(expr, self.config.type, self.vocab, self.config)

    def list_to_data(self, data_list):
        if self.config.source == 'mongodb' and not self.config.data_options['dict_in_memory']:
            tmp_expr_dict = {v["_id"]: self.to_data({x: v["data"][x] for x in self.config.data_options['filter']})
                             for v in self.expr_col.find({'_id': {'$in': data_list}})}
            batch = [tmp_expr_dict[d] for d in data_list]
        else:
            batch = [self.expr_dict[d] for d in data_list]

        return list_to_data(batch, config=self.config)

    def gen_batch(self, batch):
        # todo filter negative sampling to be disjoint from positive samples
        # batch will be a list of proof step dictionaries with goal, thms, tactic_id

        goals = [x['goal'] for x in batch]
        goals = self.list_to_data(goals)

        pos_thms = [random.choice(x['thms']) if len(x['thms']) > 0
                    else 'NO_PARAM'
                    for x in batch]

        pos_thms = self.list_to_data(pos_thms)

        tacs = torch.LongTensor([x['tac_id'] for x in batch])

        # random negative samples per goal
        neg_thms = [[a for a in random.sample(self.thms_ls, self.batch_size - 1)] for _ in range(self.batch_size)]

        neg_thms = [self.list_to_data(th) for th in neg_thms]

        return goals, tacs, pos_thms, neg_thms
