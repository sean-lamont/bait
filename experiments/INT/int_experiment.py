import json
import logging
import os
import pickle
import random

import hydra
import lightning.pytorch as pl
import torch
import torch.utils.data as data_handler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from environments.int_environment.algos.eval import eval_agent
from environments.int_environment.algos.lib.obs import nodename2index, thm2index, batch_process
from environments.int_environment.algos.model.thm_model import ThmNet
from environments.int_environment.algos.model.thm_model_transformer import ThmNet as TransThmNet
from environments.int_environment.data_generation.generate_problems import generate_multiple_problems
from environments.int_environment.data_generation.utils import Dataset


def config_to_dict(conf):
    return OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )


def load_data(data_dir, mode="train"):
    file_name = os.path.join(data_dir, '{}.pkl'.format(mode))
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def load_all_data(train_dirs, test_dirs):
    train_dataset = Dataset([])
    val_dataset = Dataset([])
    train_first_dataset = Dataset([])
    val_first_dataset = Dataset([])
    for train_dir in train_dirs:
        train_ds = load_data(train_dir, mode="train")
        train_dataset.merge(train_ds)
        val_ds = load_data(train_dir, mode="val")
        val_dataset.merge(val_ds)
        train_first_ds = load_data(train_dir, mode="train_first")
        train_first_dataset.merge(train_first_ds)
        val_first_ds = load_data(train_dir, mode="val_first")
        val_first_dataset.merge(val_first_ds)

    test_dataset = Dataset([])
    test_first_dataset = Dataset([])
    for test_dir in test_dirs:
        test_ds = load_data(test_dir, mode="test")
        test_dataset.merge(test_ds)
        test_first_ds = load_data(test_dir, mode="test_first")
        test_first_dataset.merge(test_first_ds)

    return {
        "train": train_dataset,
        "train_first": train_first_dataset,
        "val": val_dataset,
        "val_first": val_first_dataset,
        "test": test_dataset,
        "test_first": test_first_dataset
    }


class INTDataModule(pl.LightningDataModule):
    def __init__(self, config):
        self.config = config

        super().__init__()

        # all_first datasets are only the first step in the proof, and are used to initialise the rollout in evaluation
        if not self.config.online:
            train_dirs = [os.path.join(self.config.path_to_data, train_dir) for train_dir in config.train_sets]
            test_dirs = [os.path.join(self.config.path_to_data, test_dir) for test_dir in config.test_sets]
            all_data = load_all_data(train_dirs, test_dirs)

            (self.train_dataset, self.val_dataset, self.eval_dataset, self.train_first_dataset,
             self.val_first_dataset, self.eval_first_dataset) = (all_data["train"], all_data["val"], all_data["test"],
                                                                 all_data["train_first"], all_data["val_first"],
                                                                 all_data["test_first"])
        else:
            if self.config.online_order_generation:
                self.kl_dict = json.load(open(os.path.join(self.config.combo_path, "combinations.json"), "r"))
            else:
                self.kl_dict = json.load(open(os.path.join(self.config.combo_path, "orders.json"), "r"))

            self.val_dataset = Dataset([])
            self.eval_dataset = Dataset([])
            self.eval_first_dataset = Dataset([])

            for kl in self.config.test_sets:
                k = kl.split("_")[0][-1]
                l = int(kl[-1])

                data_path = os.path.join(self.config.combo_path,
                                         "test_first_dataset_prob{}_k{}l{}_oog{}_nooc{}_degree{}.pkl".format(
                                             self.config.num_probs, k, l,
                                             self.config.online_order_generation, self.config.num_order_or_combo,
                                             self.config.degree)
                                         )

                if os.path.isfile(data_path):
                    with pickle.load(open(data_path, "rb")) as existent_dataset:
                        self.eval_first_dataset.merge(existent_dataset)
                else:
                    if self.config.online_order_generation:
                        keyword_arguments = {"combos": self.kl_dict}
                    else:
                        keyword_arguments = {"orders": self.kl_dict}

                    one_piece_of_data, _ = generate_multiple_problems(k, l, num_probs=self.config.num_probs,
                                                                      train_test="test", backwards=True,
                                                                      transform_gt=self.config.transform_gt,
                                                                      degree=self.config.degree,
                                                                      num_order_or_combo=self.config.num_order_or_combo,
                                                                      **keyword_arguments)

                    self.eval_dataset.merge(one_piece_of_data["all"])
                    self.eval_first_dataset.merge(one_piece_of_data["all_first"])

            self.eval_objectives = set([problem[0]["objectives"][0].name for problem in self.eval_first_dataset])

            print("Eval dataset length ", len(self.eval_dataset))
            print("Eval first step dataset length ", len(self.eval_first_dataset))
            self.reset()

        # Every epoch this is checked, simplified with DataModule by reloading

    def collate(self, batch):
        if self.config.obs_mode == 'geometric':
            return batch_process(batch)
        else:
            return batch_process(batch, mode='seq')

    def reset(self):
        self.train_dataset = Dataset([])
        self.train_first_dataset = Dataset([])
        for kl in self.config.train_sets:
            k = kl.split("_")[0][-1]
            l = int(kl[-1])

            if self.config.online_order_generation:
                keyword_arguments = {"combos": self.kl_dict}
            else:
                keyword_arguments = {"orders": self.kl_dict}

            one_piece_of_data, problems = generate_multiple_problems(k, l, num_probs=self.config.num_probs,
                                                                     train_test="train", backwards=True,
                                                                     transform_gt=self.config.transform_gt,
                                                                     degree=self.config.degree,
                                                                     num_order_or_combo=self.config.num_order_or_combo,
                                                                     avoid_objective_names=self.eval_objectives,
                                                                     **keyword_arguments)

            # all_first used to initialise rollouts
            self.train_dataset.merge(one_piece_of_data["all"])
            self.train_first_dataset.merge(one_piece_of_data["all_first"])

    def train_dataloader(self):
        # sampler = data_handler.RandomSampler(self.train_dataset)
        # batcher = data_handler.BatchSampler(sampler, batch_size=self.config.batch_size, drop_last=False)
        # batch = self.train_dataset.get_multiple(indices=indices)
        # batch_states, batch_actions, batch_name_actions = batch_process(batch, mode=self.config.obs_mode)
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, collate_fn=self.collate)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.config.batch_size, collate_fn=self.collate)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.config.batch_size, collate_fn=self.collate)


class INTLoop(pl.LightningModule):
    def __init__(self,
                 thm_net,
                 data_module,
                 config):

        super().__init__()

        self.config = config
        self.thm_net = thm_net
        self.lr = config.optimiser_config.learning_rate
        self.batch_size = config.batch_size
        self.data_module = data_module

    def on_train_start(self):
        train_first_success_rate, train_first_wrong_case, train_first_right_case, train_first_avg_proof_length = \
            self.test_rollout(self.data_module.train_first_dataset)

        self.log_dict({"train_first_success_rate": train_first_success_rate,
                       "train_first_avg_proof_length": train_first_avg_proof_length}, batch_size=self.config.batch_size)

    def forward(self, batch_states, batch_actions, sl_train=True):
        return self.thm_net(batch_states, batch_actions, sl_train)

    def training_step(self, batch, batch_idx):
        batch_states, batch_actions, batch_name_actions = batch

        log_probs, _, _, (
            lemma_acc, ent_acc, name_acc, diff_lemma_indices, diff_ent_lemma_indices) = self.forward(
            batch_states, batch_actions)

        loss = -log_probs.mean()

        self.log_dict({'loss': loss.detach(),
                       'lemma_acc': lemma_acc.detach(),
                       'ent_acc': ent_acc.detach(),
                       'name_acc': name_acc.detach()}, batch_size=self.config.batch_size)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_states, batch_actions, batch_name_actions = batch

        log_probs, _, _, \
            (lemma_acc, ent_acc, name_acc, different_lemma_indices, different_ent_lemma_indices) = \
            self.forward(
                batch_states, batch_actions, sl_train=True
            )

        loss = -log_probs.mean()

        self.log_dict({'val_loss': loss.detach(),
                       'val_lemma_acc': lemma_acc.detach(),
                       'val_ent_acc': ent_acc.detach(),
                       'val_name_acc': name_acc.detach()}, batch_size=self.config.batch_size)

        return

    def test_rollout(self, dataset, full_dataset=False):
        self.eval()
        if full_dataset:
            eval_data = dataset.trajectories
        else:
            indices = range(len(dataset))
            eval_data = [dataset.trajectories[index] for index in random.sample(indices, k=self.config.num_test_probs)]

        env_config = {
            "mode": "eval",
            "eval_dataset": eval_data,
            "online": False,
            "batch_eval": False,
            "verbo": True,
            "obs_mode": self.config.obs_mode,
            "bag_of_words": self.config.bag_of_words,
            "time_limit": self.config.time_limit,
            "degree": self.config.degree
        }

        success_rate, wrong_cases, success_cases, avg_num_steps = \
            eval_agent(self.thm_net, env_config=env_config)

        self.train()
        return success_rate, wrong_cases, success_cases, avg_num_steps

    def on_train_epoch_end(self):
        if self.current_epoch % self.config.epochs_per_case_record == 0:
            logging.info("Testing success rate on current proofs..")

            # Test success rate after training
            train_first_success_rate, train_first_wrong_case, train_first_right_case, train_first_avg_proof_length = \
                self.test_rollout(self.data_module.train_first_dataset)

            self.log_dict({"train_first_success_rate": train_first_success_rate,
                           "train_first_avg_proof_length": train_first_avg_proof_length},
                          batch_size=self.config.batch_size)

            # val_first_success_rate, val_first_wrong_case, val_first_right_case, val_first_avg_proof_length = \
            #     self.test_rollout(self.data_module.val_first_dataset)

            # self.log_dict({"val_first_success_rate": val_first_success_rate,
            #                "val_first_avg_proof_length": val_first_avg_proof_length})

            test_first_success_rate, test_first_wrong_case, test_first_right_case, test_first_avg_proof_length = \
                self.test_rollout(self.data_module.eval_first_dataset, full_dataset=True)

            self.log_dict({"test_first_success_rate": test_first_success_rate,
                           "test_first_avg_proof_length": test_first_avg_proof_length},
                          batch_size=self.config.batch_size)

            cases_record = {
                "train_first_wrong_case": train_first_wrong_case,
                "train_first_right_case": train_first_right_case,
                # "val_first_wrong_case": val_first_wrong_case,
                # "val_first_right_case": val_first_right_case,
                "test_first_wrong_case": test_first_wrong_case,
                "test_first_right_case": test_first_right_case
            }

            json.dump(cases_record,
                      open(
                          os.path.join(
                              self.config.dump,
                              # str(self.config.timestamp),
                              f"cases_record{(int(self.global_step / self.config.updates))}.json"),
                          "w")
                      )

        if self.current_epoch % self.config.epochs_per_online_dataset == 0:
            logging.info("Generating new proofs..")
            self.data_module.reset()

            if self.current_epoch % self.config.epochs_per_new_dataset_eval == 0:
                train_first_success_rate, train_first_wrong_case, train_first_right_case, train_first_avg_proof_length = \
                    self.test_rollout(self.data_module.train_first_dataset)

                self.log_dict({"new_dataset_success_rates": train_first_success_rate,
                               "new_dataset_avg_proof_lengths": train_first_avg_proof_length},
                              batch_size=self.config.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def backward(self, loss, *args, **kwargs) -> None:
        try:
            loss.backward()
        except Exception as e:
            print(f"Error in backward: {e}")


@hydra.main(config_path="configs/experiments", config_name="int_base")
def int_experiment(config):
    os.makedirs(config.exp_config.checkpoint_dir, exist_ok=True)
    config.dump = os.path.join(config.exp_config.directory, config.dump)
    os.makedirs(config.dump)

    torch.set_float32_matmul_precision('high')

    if config.num_order_or_combo < 0:
        config.num_order_or_combo = None

    options = dict(
        num_nodes=len(nodename2index),
        num_lemmas=len(thm2index),
        hidden_dim=config.hidden_dim,
        gnn_type=config.gnn_type,
        combined_gt_obj=config.combined_gt_obj,
        attention_type=config.atten_type,
        hidden_layers=config.hidden,
        norm=config.norm,
        entity_cost=config.entity_cost,
        lemma_cost=config.lemma_cost,
        cuda=config.use_gpu,
        attention_heads=config.attention_heads,
        gat_dropout_rate=config.gat_dropout_rate,
        dropout_rate=config.dropout_rate,
    )

    data_module = INTDataModule(config)

    if config.obs_mode == 'geometric':
        experiment = INTLoop(ThmNet(**options),
                             data_module=data_module,
                             config=config)
    else:
        experiment = INTLoop(TransThmNet(**options),
                             data_module=data_module,
                             config=config)

    if config.exp_config.resume:
        print('resuming')
        logger = WandbLogger(project=config.logging_config.project,
                             name=config.exp_config.name,
                             # config=config_to_dict(config),
                             notes=config.logging_config.notes,
                             offline=config.logging_config.offline,
                             save_dir=config.exp_config.directory,
                             id=config.logging_config.id,
                             resume='must',
                             )

    else:
        logger = WandbLogger(project=config.logging_config.project,
                             name=config.exp_config.name,
                             # config=config_to_dict(config),
                             notes=config.logging_config.notes,
                             offline=config.logging_config.offline,
                             save_dir=config.exp_config.directory,
                             )

    callbacks = []

    checkpoint_callback = ModelCheckpoint(monitor="val_ent_acc", mode="max",
                                          auto_insert_metric_name=True,
                                          save_top_k=3,
                                          filename="{epoch}-{val_ent_acc}",
                                          save_last=True,
                                          every_n_epochs=10,
                                          dirpath=config.exp_config.checkpoint_dir)

    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        check_val_every_n_epoch=10,
        max_epochs=config.epochs,
        logger=logger,
        enable_progress_bar=True,
        callbacks=callbacks,
        accelerator=config.exp_config.accelerator,
        devices=config.exp_config.device
    )

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

    # trainer.fit(model=experiment, datamodule=data_module)
    logger.experiment.finish()


if __name__ == '__main__':
    int_experiment()
