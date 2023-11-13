import itertools
from itertools import islice

import lightning
import lightning.pytorch as pl
import numpy as np
import ray
import torch
from lightning import Trainer
from pymongo import MongoClient
from tqdm import tqdm

from data.stream_dataset import MongoStreamDataset, CursorIter


class MongoModule(lightning.LightningDataModule):
    def __init__(self, ds):
        super().__init__()
        # pl.seed_everything(1)
        self.ds_ = ds

    def setup(self, stage: str) -> None:
        self.ds = self.ds_.split(self.trainer.world_size, equal=True)[self.trainer.global_rank]
        # self.ds = self.ds_

    def train_dataloader(self):
        return self.ds.iter_torch_batches(collate_fn=self.collate_fn)

    def val_dataloader(self):
        # print (self.trainer.node_rank, self.trainer.global_rank, self.trainer.local_rank, self.trainer.world_size)
        # it = islice(self.ds.iter_torch_batches(collate_fn=self.collate_fn), self.trainer.local_rank, None, self.trainer.world_size + 1)
        it = self.ds.iter_torch_batches(collate_fn=self.collate_fn)
        print (next(it))
        return it

    def collate_fn(self, batch):
        return (np.copy(batch['goal']), torch.from_numpy(np.copy(batch['target'])))
        # return (np.copy(batch['goal']), np.copy(batch['tactic']), torch.from_numpy(np.copy(batch['distance_to_proof'])))


class TestModule(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.nn = torch.nn.Linear(128, 128)

    def forward(self, batch):
        return torch.sum(batch[2])

    def training_step(self, batch):
        self.log(self.global_step, prog_bar=True)
        return self(batch)

    def validation_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


if __name__ == '__main__':
    client = MongoClient()

    db = client['lean_dojo']
    collection = 'goal_len_task'
    col = db[collection]

    data = []

    goal_fields = {"_id": 0, "goal": 1, "target": 1}

    ds = ray.data.read_mongo(
        uri='localhost:27017',
        database='lean_dojo',
        collection=collection,
        # sort by random index so data is shuffled, and constant between cursors
        pipeline=[{'$match': {'rand_idx': {'$lt': 0.9}}},
                  {"$sort": {'rand_idx': 1}},
                  {'$project': goal_fields},
                  ],

    )

    # tac_fields = {"_id": 0, "goal": 1, "winner": 1, "winner_prob": 1, "loser": 1, "loser_prob": 1}

    data_module = MongoModule(ds)

    # data_module = MongoModulev2('lean_dojo', 'edge_data', None)

    model = TestModule()

    trainer = Trainer()

    trainer.validate(model, data_module)
    # trainer.validate(model, data_module)

    # trainer.validate(model, data_module)
    # val_loader = data_module.val_dataloader()

    # data = []
    # for d in tqdm(val_loader):
    #     data.append(d)

# class MongoModulev2(lightning.LightningDataModule):
#     def __init__(self, db, col, query, uri='localhost:27017'):
#         super().__init__()
#
#         pl.seed_everything(1)
#         client = MongoClient()
#         self.ds = client[db][col]
#
#         # sort by random index so data is shuffled, and constant between cursors
#         self.cursor = self.ds.aggregate(
#             [{"$sort": {'rand_idx': 1}},
#              {"$project": {"_id": 0, "goal": 1, "tactic": 1, "distance_to_proof": 1}}])
#
#         self.ds = itertools.cycle(CursorIter(cursor=self.cursor, fields=['goal', 'tactic', 'distance_to_proof']))
#
#     def train_dataloader(self):
#         return islice(self.cursor, self.trainer.local_rank, None, self.trainer.world_size + 1)
#         # return self.ds.iter_torch_batches(collate_fn=self.collate_fn)
#
#     def val_dataloader(self):
#         # print (self.trainer.node_rank, self.trainer.global_rank, self.trainer.local_rank, self.trainer.world_size)
#         return islice(self.cursor, self.trainer.local_rank, None, self.trainer.world_size + 1)
#
#     def collate_fn(self, batch):
#         goals = [g['goal'] for g in batch]
#         tactics = [g['tactic'] for g in batch]
#         y = [g['distance_to_proof'] for g in batch]
#         return (goals, tactics, y)
#
