import logging
import os

import lightning.pytorch as pl
import pyrallis
import torch

from data.get_data import get_data
from experiments.pyrallis_configs_old import HOListPretrainConfig
from models.get_model import get_model
from models.holist_models.tactic_predictor import TacticPrecdictor, CombinerNetwork
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from experiments.holist.train import torch_training_module


def config_to_dict(conf):
    return {
        k: config_to_dict(v) if hasattr(v, '__dict__') else v
        for k, v in vars(conf).items()
    }

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.exp_config.directory)
        os.makedirs(self.config.checkpoint_dir)

    def run(self):
        # logging.basicConfig(level=logging.DEBUG)

        torch.set_float32_matmul_precision('high')
        experiment = torch_training_module.HOListTraining_(embedding_model_goal=get_model(self.config.model_config),
                                                                embedding_model_premise=get_model(self.config.model_config),
                                                                tac_model=TacticPrecdictor(
                                                                    embedding_dim=self.config.final_embed_dim,
                                                                    num_tactics=self.config.num_tactics),
                                                                combiner_model=CombinerNetwork(
                                                                    embedding_dim=self.config.final_embed_dim,
                                                                    num_tactics=self.config.num_tactics,
                                                                    tac_embed_dim=self.config.tac_embed_dim),
                                                                lr=self.config.optimiser_config.learning_rate,
                                                                batch_size=self.config.batch_size)

        data_module = get_data(self.config.data_config, experiment='holist_pretrain')

        logger = WandbLogger(project=self.config.exp_config.logging_config.project,
                             name=self.config.exp_config.name,
                             config=config_to_dict(self.config),
                             notes=self.config.exp_config.logging_config.notes,
                             offline=self.config.exp_config.logging_config.offline,
                             save_dir=self.config.exp_config.directory,
                             )

        callbacks = []

        checkpoint_callback = ModelCheckpoint(monitor="rel_param_acc", mode="max",
                                              auto_insert_metric_name=True,
                                              save_top_k=3,
                                              filename="{epoch}-{rel_param_acc}-{topk_acc}",
                                              save_on_train_epoch_end=True,
                                              save_last=True,
                                              save_weights_only=True,
                                              dirpath=self.config.checkpoint_dir)

        callbacks.append(checkpoint_callback)

        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=500,
            callbacks=callbacks,
            accelerator=self.config.exp_config.accelerator,
            devices=self.config.exp_config.device
        )


        trainer.val_check_interval = self.config.val_frequency
        if self.config.limit_val_batches:
            trainer.limit_val_batches = self.config.val_size // self.config.batch_size

        trainer.fit(model=experiment, datamodule=data_module)
        logger.experiment.finish()


def main():
    cfg = pyrallis.parse(config_class=HOListPretrainConfig)
    experiment = ExperimentRunner(cfg)
    experiment.run()

if __name__ == '__main__':
    main()
