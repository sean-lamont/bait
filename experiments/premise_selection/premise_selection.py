import logging
import os
import warnings

import hydra
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from data.get_data import get_data

warnings.filterwarnings('ignore')
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from models.get_model import get_model
from models.gnn.formula_net.formula_net import BinaryClassifier
import torch


def config_to_dict(conf):
    return OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )


def binary_loss(preds, targets):
    return -1. * torch.mean(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))


class PremiseSelection(pl.LightningModule):
    def __init__(self,
                 embedding_model_goal,
                 embedding_model_premise,
                 classifier,
                 batch_size=32,
                 lr=1e-4):
        super().__init__()

        self.embedding_model_goal = embedding_model_goal
        self.embedding_model_premise = embedding_model_premise
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


'''

Premise selection experiment with separate encoders for goal and premise

'''


@hydra.main(config_path="../../configs/experiments")#, config_name="experiments/ho")
def premise_selection_experiment(config):
    torch.set_float32_matmul_precision('medium')

    OmegaConf.resolve(config)
    os.makedirs(config.exp_config.directory + '/checkpoints', exist_ok=True)

    experiment = PremiseSelection(get_model(config.model_config),
                                  get_model(config.model_config),
                                  BinaryClassifier(config.model_config.model_attributes['embedding_dim'] * 2),
                                  lr=config.optimiser_config.learning_rate,
                                  batch_size=config.batch_size)

    data_module = get_data(config.data_config)

    if config.exp_config.resume:
        logging.info('Resuming run..')
        logger = WandbLogger(project=config.logging_config.project,
                             name=config.exp_config.name,
                             config=config_to_dict(config),
                             notes=config.logging_config.notes,
                             offline=config.logging_config.offline,
                             save_dir=config.exp_config.directory,
                             id=config.logging_config.id,
                             resume='must',
                             )

    else:
        logger = WandbLogger(project=config.logging_config.project,
                             name=config.exp_config.name,
                             config=config_to_dict(config),
                             notes=config.logging_config.notes,
                             offline=config.logging_config.offline,
                             save_dir=config.exp_config.directory,
                             )


    callbacks = []

    checkpoint_callback = ModelCheckpoint(monitor="acc", mode="max",
                                          auto_insert_metric_name=True,
                                          save_top_k=3,
                                          filename="{epoch}-{acc}",
                                          save_last=True,
                                          dirpath=config.exp_config.checkpoint_dir)

    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=500,
        callbacks=callbacks,
        accelerator=config.exp_config.accelerator,
        devices=config.exp_config.device
    )

    trainer.val_check_interval = config.val_frequency

    if config.limit_val_batches:
        trainer.limit_val_batches = config.val_size // config.batch_size


    if config.exp_config.resume:

        logging.debug("Resuming experiment from last checkpoint..")
        ckpt_dir = config.exp_config.checkpoint_dir + "/last.ckpt"

        if not os.path.exists(ckpt_dir):
            raise Exception(f"Missing checkpoint in {ckpt_dir}")

        logging.debug("Resuming experiment from last checkpoint..")
        ckpt_dir = config.exp_config.checkpoint_dir + "/last.ckpt"
        state_dict = torch.load(ckpt_dir)['state_dict']

        # if hasattr(config, 'test_checkpoint'):
        if config.test_checkpoint:
            logging.info("Running checkpoint on test dataset..")
            trainer.test(model=experiment, datamodule=data_module, ckpt_path=ckpt_dir)
        else:
            experiment.load_state_dict(state_dict)
            trainer.fit(model=experiment, datamodule=data_module, ckpt_path=ckpt_dir)
    else:
        trainer.fit(model=experiment, datamodule=data_module)
        # if hasattr(config, 'test_on_finish'):
        if config.test_on_finish:
            logging.info("Testing best model from run..")
            trainer.test(model=experiment, datamodule=data_module)


    logger.experiment.finish()


if __name__ == '__main__':
    premise_selection_experiment()

# @hydra.main(config_path="configs/", config_name="formula_net_hol4")
# def run_experiment(cfg):
#     (cfg)
# print(OmegaConf.to_yaml(cfg.exp_config))

# wandb.config = OmegaConf.to_container(
#     cfg, resolve=True, throw_on_missing=True
# )
# wandb.init(project=cfg.wandb.project)
# wandb.log({"loss": loss})
# model = Model(**wandb.config.model.configs)

# run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
# wandb.log({"loss": loss})

# if __name__ == '__main__':
#     run_experiment()
# if __name__ == '__main__':
# sweep_config = {
#     'method': 'random',
#     'name': 'first_sweep',
#     'metric': {
#         'goal': 'minimize',
#         'name': 'validation_loss'
#     },
#     'parameters': {
#         'n_hidden': {'values': [2, 3, 5, 10]},
#         'lr': {'max': 1.0, 'min': 0.0001},
#         'noise': {'max': 1.0, 'min': 0.}
#     }
# }
#
# sweep_id = wandb.sweep(sweep_config, project="test_sweep")
# wandb.agent(sweep_id=sweep_id, function=run_experiment, count=5)


# def main():
#     cfg = pyrallis.parse(config_class=PremiseSelectionConfig)
#     experiment = SeparateEncoderPremiseSelection(cfg)
#     experiment.run()
#
# if __name__ == '__main__':
#     main()
