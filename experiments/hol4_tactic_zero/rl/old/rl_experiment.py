from experiments.hol4_tactic_zero.rl.old.lightning_rl import TacticZeroLoop
from models.get_model import get_model
from lightning.pytorch.callbacks import ModelCheckpoint
from experiments.hol4_tactic_zero.rl.old.agent_utils import *
from lightning.pytorch.loggers import WandbLogger
from experiments.hol4_tactic_zero.rl.old.rl_data_module import *
from models.tactic_zero.policy_models import ArgPolicy, TacPolicy, TermPolicy, ContextPolicy
from models.gnn.formula_net.formula_net import FormulaNetEdges
import lightning.pytorch as pl
import torch.optim
from environments.hol4.new_env import *
import warnings

warnings.filterwarnings('ignore')

def get_model_dict(prefix, state_dict):
    return {k[len(prefix) + 1:]: v for k, v in state_dict.items()
            if k.startswith(prefix)}


# hack for now to deal with BatchNorm Loading
def get_model_dict_fn(model, prefix, state_dict):
    ret_dict = {}
    own_state = model.state_dict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k[len(prefix) + 1:]
            if "mlp.3" in k:
                k = k.replace('3', '2')
            if k not in own_state:
                continue
            ret_dict[k] = v
    return ret_dict


def get_model_sat(prefix, state_dict):
    ret_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            if 'complete_edge_index' not in k:
                k = k[len(prefix) + 1:]
                ret_dict[k] = v
    return ret_dict


class RLExperiment:
    def __init__(self, config):
        self.config = config

    def load_pretrained_encoders(self, encoder_premise, encoder_goal):
        ckpt_dir = self.config['pretrain_ckpt']
        ckpt = torch.load(ckpt_dir)['state_dict']

        if self.config['exp_type'] == 'gnn':
            encoder_premise.load_state_dict(get_model_dict_fn(encoder_premise, 'embedding_model_premise', ckpt))
            encoder_goal.load_state_dict(get_model_dict_fn(encoder_goal, 'embedding_model_goal', ckpt))
        elif self.config['exp_type'] == 'sat':
            encoder_premise.load_state_dict(get_model_sat('embedding_model_premise', ckpt))
            encoder_goal.load_state_dict(get_model_sat('embedding_model_goal', ckpt))
        else:
            encoder_premise.load_state_dict(get_model_dict('embedding_model_premise', ckpt))
            encoder_goal.load_state_dict(get_model_dict('embedding_model_goal', ckpt))

    def run(self):
        VOCAB_SIZE = self.config['vocab_size']
        EMBEDDING_DIM = self.config['embedding_dim']

        resume = self.config['resume']
        pretrain = self.config['pretrain']

        tactics = self.config['tactics']
        tactic_pool = tactics['tactic_pool']

        # default policy models
        context_net = ContextPolicy()
        tac_net = TacPolicy(len(tactic_pool))
        arg_net = ArgPolicy(len(tactic_pool), EMBEDDING_DIM)
        term_net = TermPolicy(len(tactic_pool), EMBEDDING_DIM)
        induct_net = FormulaNetEdges(VOCAB_SIZE, EMBEDDING_DIM, self.config['gnn_layers'], global_pool=False,
                                     batch_norm=False)

        # encoder_premise = get_model(self.config['model_config'])
        # encoder_goal = get_model(self.config['model_config'])


        encoder_premise = FormulaNetEdges(input_shape=1004,
                               embedding_dim=256,
                               num_iterations=1,
                        batch_norm=False)

        encoder_goal = FormulaNetEdges(input_shape=1004,
                        embedding_dim=256,
                        num_iterations=1,
                        batch_norm=False)

        notes = self.config['notes']
        save_dir = self.config['dir'] + notes

        experiment_config = {'max_steps': self.config['max_steps'],
                             'gamma': self.config['gamma'],
                             'lr': self.config['lr'],
                             'arg_len': self.config['arg_len'],
                             'data_type': self.config['data_type'],
                             'dir_path': save_dir}

        if pretrain:
            print ("Loading pretrained encoder models..")
            self.load_pretrained_encoders(encoder_premise, encoder_goal)

        if resume:
            experiment_config['replay_dir'] = save_dir + 'replays'

            logger = WandbLogger(project=self.config["project"],
                                 name=self.config['name'],
                                 config=experiment_config,
                                 notes=notes,
                                 id=self.config['resume_id'],
                                 resume='must',
                                 # offline=True,
                                 )
        else:
            logger = WandbLogger(project=self.config["project"],
                                 name=self.config['name'],
                                 config=experiment_config,
                                 notes=notes,
                                 # offline=True,
                                 )



        module = RLData(train_goals=self.config['train_goals'], test_goals=self.config['test_goals'], database=self.config['database'],
                        graph_db=self.config['graph_db'],
                        config=self.config)

        experiment = TacticZeroLoop(context_net=context_net,
                                    tac_net=tac_net,
                                    arg_net=arg_net,
                                    term_net=term_net,
                                    induct_net=induct_net,
                                    encoder_premise=encoder_premise,
                                    encoder_goal=encoder_goal,
                                    config=experiment_config,
                                    graph_db=self.config['graph_db'],
                                    token_enc=self.config['token_enc'],
                                    tactics=tactics, reverse_database=self.config['reverse_database'])
        callbacks = []

        checkpoint_callback = ModelCheckpoint(monitor="val_proven", mode="max",
                                              auto_insert_metric_name=True,
                                              save_top_k=3,
                                              filename="{epoch}-{val_proven}-{cumulative_proven}",
                                              save_on_train_epoch_end=True,
                                              save_last=True,
                                              dirpath=save_dir)

        callbacks.append(checkpoint_callback)

        trainer = pl.Trainer(devices=self.config['device'],
                             check_val_every_n_epoch=1000,
                             # check_val_every_n_epoch=self.config['val_freq'],
                             logger=logger,
                             callbacks=callbacks,

                             # max_steps=10,
                             )

        if resume and os.path.exists(save_dir + "last.ckpt"):
            print ("Resuming experiment from last checkpoint..")
            ckpt_dir = save_dir + "last.ckpt"
            state_dict = load_state(torch.load(ckpt_dir)['state_dict'], experiment)
            ckpt = torch.load(ckpt_dir)
            new_dict = {k:v for k,v in ckpt.items() if k != 'state_dict'}
            new_dict['state_dict'] = state_dict
            torch.save(new_dict, ckpt_dir)

            experiment.load_state_dict(state_dict)
            trainer.fit(experiment, module, ckpt_path=ckpt_dir)
        else:
            trainer.fit(experiment, module)


def load_state(state_dict, experiment):
    own_dict = experiment.state_dict()
    ret_dict = {}
    for k,v in state_dict.items():
        if k not in own_dict:
            continue
        ret_dict[k] = v
    return ret_dict
