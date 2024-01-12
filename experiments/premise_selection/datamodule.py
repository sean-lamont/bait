import pickle

import torch
from lightning.pytorch import LightningDataModule
from pymongo import MongoClient
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from data.utils.stream_dataset import MongoStreamDataset
from data.utils.graph_data_utils import transform_expr, transform_batch


class PremiseDataModule(LightningDataModule):
    def __init__(self, config):

        super().__init__()
        self.config = config

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

            # if dict_in_memory save all expressions to disk, otherwise return cursor
            if self.config.data_options['dict_in_memory']:
                self.expr_dict = {v["_id"]: self.to_data({x: v["data"][x] for x in self.config.data_options['filter']})
                                  for v in tqdm(expr_col.find({}))}
            else:
                self.expr_col = expr_col

            fields = ['conj', 'stmt', 'y']

            # if data_in_memory, save all examples to disk
            if self.config.data_options['split_in_memory']:
                self.train_data = [{field: v[field] for field in fields}
                                   for v in tqdm(split_col.find({'split': 'train'}))]

                self.val_data = [{field: v[field] for field in fields}
                                 for v in tqdm(split_col.find({'split': 'val'}))]

                self.test_data = [{field: v[field] for field in fields}
                                  for v in tqdm(split_col.find({'split': 'test'}))]

            # stream dataset from MongoDB
            else:
                train_len = split_col.count_documents({'split': 'train'})
                val_len = split_col.count_documents({'split': 'val'})
                test_len = split_col.count_documents({'split': 'test'})

                self.train_data = MongoStreamDataset(split_col.find({'split': 'train'}), len=train_len,
                                                     fields=fields)

                self.val_data = MongoStreamDataset(split_col.find({'split': 'val'}), len=val_len,
                                                   fields=fields)

                self.test_data = MongoStreamDataset(split_col.find({'split': 'test'}), len=test_len,
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
            self.test_data = self.data['test_data']

        else:
            raise NotImplementedError

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        if self.config.type == 'custom':
            pass
            # add batch transfers here for custom data types
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

    def list_to_data(self, data_list):
        # either stream from database when too large for memory, or we can save all to disk for quick access
        if self.config.source == 'mongodb' and not self.config.data_options['dict_in_memory']:
            tmp_expr_dict = {v["_id"]: self.to_data({x: v["data"][x] for x in self.config.data_options['filter']})
                             for v in self.expr_col.find({'_id': {'$in': data_list}})}

            batch = [tmp_expr_dict[d] for d in data_list]
        else:
            batch = [self.expr_dict[d] for d in data_list]

        return transform_batch(batch, config=self.config)

    def to_data(self, expr):
        return transform_expr(expr, self.config.type, self.vocab, self.config)

    def collate_data(self, batch):
        y = torch.LongTensor([b['y'] for b in batch])
        data_1 = self.list_to_data([b['conj'] for b in batch])
        data_2 = self.list_to_data([b['stmt'] for b in batch])
        return data_1, data_2, y

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config.batch_size,
                          collate_fn=self.collate_data, shuffle=self.config.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.config.batch_size,
                          collate_fn=self.collate_data, shuffle=self.config.shuffle)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.config.batch_size,
                          collate_fn=self.collate_data)
