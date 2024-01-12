import warnings

import hydra
import torch.optim
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from environments.hol4.env_wrapper import *
from experiments.tacticzero.hol4_tactic_zero import HOL4TacticZero
from experiments.tacticzero.tactic_zero_data_module import *
from models.get_model import get_model
from models.gnn.formula_net.formula_net import FormulaNetEdges
from models.tactic_zero.policy_models import ArgPolicy, TacPolicy, TermPolicy, ContextPolicy

warnings.filterwarnings('ignore')


def config_to_dict(conf):
    return OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )


def get_model_dict(prefix, state_dict):
    return {k[len(prefix) + 1:]: v for k, v in state_dict.items()
            if k.startswith(prefix)}


def load_pretrained_encoders(self, encoder_premise, encoder_goal):
    ckpt_dir = self.config.pretrain_ckpt
    ckpt = torch.load(ckpt_dir)['state_dict']
    encoder_premise.load_state_dict(get_model_dict('embedding_model_premise', ckpt))
    encoder_goal.load_state_dict(get_model_dict('embedding_model_goal', ckpt))


@hydra.main(config_path="../../configs/experiments", config_name="holist_premise_selection")
def tactic_zero_experiment(config):
    OmegaConf.resolve(config)

    os.makedirs(config.exp_config.directory + '/checkpoints', exist_ok=True)

    logging.basicConfig(level=logging.DEBUG)


    torch.set_float32_matmul_precision('high')

    resume = config.exp_config.resume
    pretrain = config.pretrain

    tactics = config.tactic_config

    # default policy models
    context_net = ContextPolicy()
    tac_net = TacPolicy(len(config.tactic_config.tactic_pool))
    arg_net = ArgPolicy(len(config.tactic_config.tactic_pool), config.model_config.model_attributes.embedding_dim)
    term_net = TermPolicy(len(config.tactic_config.tactic_pool), config.model_config.model_attributes.embedding_dim)

    induct_net = FormulaNetEdges(config.model_config.model_attributes.vocab_size, config.model_config.model_attributes.embedding_dim,
                                 num_iterations=3, global_pool=False,
                                 batch_norm=False)

    encoder_premise = get_model(config.model_config)
    encoder_goal = get_model(config.model_config)

    if pretrain:
        logging.info("Loading pretrained encoder models..")
        load_pretrained_encoders(encoder_premise, encoder_goal)

    if config.exp_config.resume:
        logger = WandbLogger(project=config.logging_config.project,
                             name=config.exp_config.name,
                             config=config_to_dict(config),
                             notes=config.logging_config.notes,
                             id=config.logging_config.id,
                             resume='must',
                             offline=config.logging_config.offline,
                             save_dir=config.exp_config.directory,
                             )

    else:
        logger = WandbLogger(project=config.logging_config.project,
                             name=config.exp_config.name,
                             config=config_to_dict(config),
                             notes=config.logging_config.notes,
                             save_dir=config.exp_config.directory,
                             offline=config.logging_config.offline
                             )

    module = RLData(config.data_config)

    module.prepare_data()
    module.setup(stage="fit")

    proof_db = MongoClient()
    proof_db = proof_db[config.proof_db[0]]
    proof_db = proof_db[config.proof_db[1]]

    experiment = HOL4TacticZero(goal_net=context_net,
                                tac_net=tac_net,
                                arg_net=arg_net,
                                term_net=term_net,
                                induct_net=induct_net,
                                encoder_premise=encoder_premise,
                                encoder_goal=encoder_goal,
                                config=config,
                                tactics=tactics,
                                converter=module.list_to_data, proof_db=proof_db)

    callbacks = []

    checkpoint_callback = ModelCheckpoint(monitor="val_proven", mode="max",
                                          auto_insert_metric_name=True,
                                          save_top_k=3,
                                          filename="{epoch}-{val_proven}-{cumulative_proven}",
                                          save_on_train_epoch_end=True,
                                          save_last=True,
                                          dirpath=config.exp_config.checkpoint_dir,
                                          )

    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        devices=config.exp_config.device,
        accelerator=config.exp_config.accelerator,
        check_val_every_n_epoch=config.val_freq,
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.epochs
    )

    if resume:
        logging.debug("Resuming experiment from last checkpoint..")
        ckpt_dir = config.exp_config.checkpoint_dir + "/last.ckpt"

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
        # trainer.validate(experiment, module)
        trainer.fit(experiment, module)


def load_state(state_dict, experiment):
    own_dict = experiment.state_dict()
    ret_dict = {}
    for k, v in state_dict.items():
        if k not in own_dict:
            continue
        ret_dict[k] = v
    return ret_dict


if __name__ == '__main__':
    tactic_zero_experiment()
    # logging.basicConfig(level=logging.DEBUG)
