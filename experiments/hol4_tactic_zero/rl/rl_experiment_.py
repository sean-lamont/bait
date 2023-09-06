import logging
import warnings

import lightning.pytorch as pl
import pyrallis
import torch.optim
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from environments.hol4.new_env import *
from experiments.hol4_tactic_zero.rl.agent_utils import *
from experiments.hol4_tactic_zero.rl.hol4_tactic_zero import HOL4TacticZero
from experiments.hol4_tactic_zero.rl.tactic_zero_data_module import *
from experiments.premise_selection_old import config_to_dict
from experiments.pyrallis_configs_old import TacticZeroRLConfig
from models.get_model import get_model
from models.gnn.formula_net.formula_net import FormulaNetEdges
from models.tactic_zero.policy_models import ArgPolicy, TacPolicy, TermPolicy, ContextPolicy

warnings.filterwarnings('ignore')

def get_model_dict(prefix, state_dict):
    return {k[len(prefix) + 1:]: v for k, v in state_dict.items()
            if k.startswith(prefix)}

class RLExperiment:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.exp_config.directory, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def load_pretrained_encoders(self, encoder_premise, encoder_goal):
        ckpt_dir = self.config.pretrain_ckpt
        ckpt = torch.load(ckpt_dir)['state_dict']
        encoder_premise.load_state_dict(get_model_dict('embedding_model_premise', ckpt))
        encoder_goal.load_state_dict(get_model_dict('embedding_model_goal', ckpt))

    def run(self):
        logging.basicConfig(filename=self.config.exp_config.directory + '/log', level=logging.DEBUG)
        torch.set_float32_matmul_precision('high')

        resume = self.config.exp_config.resume
        pretrain = self.config.pretrain

        tactics = self.config.tactic_config

        # todo get non-encoder models from config

        # default policy models
        context_net = ContextPolicy()
        tac_net = TacPolicy(len(self.config.tactic_config.tactic_pool))
        arg_net = ArgPolicy(len(self.config.tactic_config.tactic_pool), self.config.model_config.embedding_dim)
        term_net = TermPolicy(len(self.config.tactic_config.tactic_pool), self.config.model_config.embedding_dim)
        induct_net = FormulaNetEdges(self.config.model_config.vocab_size, self.config.model_config.embedding_dim,
                                     num_iterations=3, global_pool=False,
                                     batch_norm=False)

        encoder_premise = get_model(self.config.model_config)
        encoder_goal = get_model(self.config.model_config)

        if pretrain:
            logging.info("Loading pretrained encoder models..")
            self.load_pretrained_encoders(encoder_premise, encoder_goal)

        if self.config.exp_config.resume:
            logger = WandbLogger(project=self.config.exp_config.logging_config.project,
                                 name=self.config.exp_config.name,
                                 config=config_to_dict(self.config),
                                 notes=self.config.exp_config.logging_config.notes,
                                 id=self.config.exp_config.logging_config.id,
                                 resume='must',
                                 offline=self.config.exp_config.logging_config.offline,
                                 save_dir=self.config.exp_config.directory,
                                 )

        else:
            logger = WandbLogger(project=self.config.exp_config.logging_config.project,
                                 name=self.config.exp_config.name,
                                 config=config_to_dict(self.config),
                                 notes=self.config.exp_config.logging_config.notes,
                                 save_dir=self.config.exp_config.directory,
                                 offline=self.config.exp_config.logging_config.offline
                                 )


        module = RLData(self.config.data_config)

        module.prepare_data()
        module.setup(stage="fit")

        proof_db = MongoClient()
        proof_db = proof_db[self.config.proof_db[0]]
        proof_db = proof_db[self.config.proof_db[1]]

        experiment = HOL4TacticZero(goal_net=context_net,
                                    tac_net=tac_net,
                                    arg_net=arg_net,
                                    term_net=term_net,
                                    induct_net=induct_net,
                                    encoder_premise=encoder_premise,
                                    encoder_goal=encoder_goal,
                                    config=self.config,
                                    tactics=tactics,
                                    converter=module.list_to_data, proof_db=proof_db)


        callbacks = []

        checkpoint_callback = ModelCheckpoint(monitor="val_proven", mode="max",
                                              auto_insert_metric_name=True,
                                              save_top_k=3,
                                              filename="{epoch}-{val_proven}-{cumulative_proven}",
                                              save_on_train_epoch_end=True,
                                              save_last=True,
                                              dirpath=self.config.checkpoint_dir,
                                              )

        callbacks.append(checkpoint_callback)

        trainer = pl.Trainer(
                            devices=self.config.exp_config.device,
                            # accelerator='cpu',
                             check_val_every_n_epoch=self.config.val_freq,
                             logger=logger,
                             callbacks=callbacks,
                             max_epochs=self.config.epochs
                             )

        if resume:
            logging.debug("Resuming experiment from last checkpoint..")
            ckpt_dir = self.config.checkpoint_dir + "/last.ckpt"

            if not os.path.exists(ckpt_dir):
                raise Exception(f"Missing checkpoint in {ckpt_dir}")

            state_dict = load_state(torch.load(ckpt_dir)['state_dict'], experiment)
            ckpt = torch.load(ckpt_dir)
            new_dict = {k: v for k, v in ckpt.items() if k != 'state_dict'}
            new_dict['state_dict'] = state_dict
            torch.save(new_dict, ckpt_dir)
            experiment.load_state_dict(state_dict)
            trainer.fit(experiment, module, ckpt_path=ckpt_dir)
        else:
            trainer.fit(experiment, module)


def load_state(state_dict, experiment):
    own_dict = experiment.state_dict()
    ret_dict = {}
    for k, v in state_dict.items():
        if k not in own_dict:
            continue
        ret_dict[k] = v
    return ret_dict

def main():
    cfg = pyrallis.parse(config_class=TacticZeroRLConfig)
    experiment = RLExperiment(cfg)
    experiment.run()

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    main()
