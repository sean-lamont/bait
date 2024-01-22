import warnings

import wandb
from omegaconf import OmegaConf

warnings.filterwarnings('ignore')
import lightning.pytorch as pl

import torch


def config_to_dict(conf):
    return OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )


def binary_loss(preds, targets):
    return -1. * torch.mean(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))


class PremiseSelection(pl.LightningModule):
    def __init__(self,
                 goal_embedding_module,
                 premise_embedding_module,
                 classifier,
                 batch_size=32,
                 lr=1e-4):
        super().__init__()

        self.embedding_model_goal = goal_embedding_module
        self.embedding_model_premise = premise_embedding_module
        self.classifier = classifier
        self.eps = 1e-6
        self.lr = lr
        self.batch_size = batch_size

    # self.save_hyperparameters()

    def forward(self, goal, premise):
        embedding_goal = self.embedding_model_goal(goal)
        embedding_premise = self.embedding_model_premise(premise)
        preds = self.classifier(torch.cat([embedding_goal, embedding_premise], dim=1))
        preds = torch.clip(preds, self.eps, 1 - self.eps)
        return torch.flatten(preds)

    def training_step(self, batch, batch_idx):
        goal, premise, y = batch
        try:
            preds = self(goal, premise)
        except Exception as e:
            print(f"Error in forward: {e}")
            return
        loss = binary_loss(preds, y)
        self.log("loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric('acc', summary='max')

        goal, premise, y = batch
        try:
            preds = self(goal, premise)
            preds = (preds > 0.5)
            acc = torch.sum(preds == y) / y.size(0)
            self.log("acc", acc, batch_size=self.batch_size, prog_bar=True)
        except Exception as e:
            print(f"Error in val forward {e}")
        return

    def test_step(self, batch, batch_idx):
        goal, premise, y = batch
        preds = self(goal, premise)
        preds = (preds > 0.5)
        acc = torch.sum(preds == y) / y.size(0)
        self.log("acc", acc, batch_size=self.batch_size, prog_bar=True)
        return

    # todo define from config
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 25], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # return torch.optim.Adam(self.parameters(), lr=self.lr)

    def backward(self, loss, *args, **kwargs) -> None:
        try:
            loss.backward()
        except Exception as e:
            print(f"Error in backward: {e}")