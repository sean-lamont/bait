from itertools import islice

import lightning
import lightning.pytorch as pl
import numpy as np
import ray
import torch
from lightning import Trainer
from pymongo import MongoClient


# todo distributed setting:
# - can add a random id field, then sort for query. Then query will be identical for all nodes, and can then do islice
#

class MongoModule(lightning.LightningDataModule):
    def __init__(self, db, col, query, uri='localhost:27017'):
        super().__init__()

        pl.seed_everything(1)

        self.ds = ray.data.read_mongo(
            uri=uri,
            database=db,
            collection=col,
            pipeline=[{"$project": {"_id": 0, "edge_attr": "$data.edge_attr"}}],
        )


    def train_dataloader(self):
        return self.ds.iter_torch_batches(collate_fn=collate_fn)

    def val_dataloader(self):
        # print (self.trainer.node_rank, self.trainer.global_rank, self.trainer.local_rank, self.trainer.world_size)
        return islice(self.ds.iter_torch_batches(collate_fn=collate_fn), self.trainer.local_rank, None, self.trainer.world_size + 1)
        # return self.ds.iter_torch_batches(collate_fn=collate_fn)

class TestModule(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.nn = torch.nn.Linear(128, 128)

    def forward(self, batch):
        return torch.sum(batch[0])

    def training_step(self, batch):
        self.log(self.global_step, prog_bar=True)
        return self(batch)

    def validation_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     print (batch)
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


if __name__ == '__main__':
    client = MongoClient()
    db = client['hol4']
    col = db['expression_graphs']

    def collate_fn(batch):
        return [torch.from_numpy(np.copy(array)) for array in batch['edge_attr']]

    data = []

    data_module = MongoModule('holstep', 'expression_graphs', None)
    model = TestModule()

    trainer = Trainer()

    trainer.validate(model, data_module)
    trainer.validate(model, data_module)
