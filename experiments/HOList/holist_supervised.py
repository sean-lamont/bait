import logging
import os

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from experiments.HOList.datamodule import HOListDataModule
from experiments.HOList.train_module import HOListTraining_
from models.get_model import get_model
from models.embedding_models.holist_models.tactic_predictor import CombinerNetwork, TacticPredictor
from utils.utils import config_to_dict


@hydra.main(config_path="configs/experiments", config_name="holist_premise_selection")
def holist_pretrain_experiment(config):
    OmegaConf.resolve(config)

    os.makedirs(config.exp_config.directory + '/checkpoints', exist_ok=True)

    torch.set_float32_matmul_precision('high')

    experiment = HOListTraining_(embedding_model_goal=get_model(config.model_config),
                                 embedding_model_premise=get_model(config.model_config),

                                 tac_model=TacticPredictor(
                                     embedding_dim=config.final_embed_dim,
                                     num_tactics=config.num_tactics),

                                 combiner_model=CombinerNetwork(
                                     embedding_dim=config.final_embed_dim,
                                     num_tactics=config.num_tactics,
                                     tac_embed_dim=config.tac_embed_dim),

                                 lr=config.optimiser_config.learning_rate,
                                 batch_size=config.data_config.batch_size)

    data_module = HOListDataModule(config=config.data_config)

    if config.exp_config.resume:
        print('resuming')
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

    checkpoint_callback = ModelCheckpoint(monitor="rel_param_acc", mode="max",
                                          auto_insert_metric_name=True,
                                          save_top_k=3,
                                          filename="{epoch}-{rel_param_acc}-{topk_acc}",
                                          save_on_train_epoch_end=True,
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
        trainer.limit_val_batches = config.val_size // config.data_config.batch_size

    if config.exp_config.resume:

        logging.debug("Resuming experiment from last checkpoint..")
        ckpt_dir = config.exp_config.checkpoint_dir + "/last.ckpt"

        if not os.path.exists(ckpt_dir):
            raise Exception(f"Missing checkpoint in {ckpt_dir}")

        logging.debug("Resuming experiment from last checkpoint..")
        ckpt_dir = config.exp_config.checkpoint_dir + "/last.ckpt"
        state_dict = torch.load(ckpt_dir)['state_dict']
        experiment.load_state_dict(state_dict)
        trainer.fit(model=experiment, datamodule=data_module, ckpt_path=ckpt_dir)
    else:
        trainer.fit(model=experiment, datamodule=data_module)

    logger.experiment.finish()


if __name__ == '__main__':
    holist_pretrain_experiment()
