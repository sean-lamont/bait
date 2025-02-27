from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import division
from __future__ import print_function
from __future__ import print_function

import warnings

import ray
import torch
from loguru import logger

from models.TacticZero.policy_models import ArgPolicy, TacPolicy, TermPolicy, ContextPolicy
from models.embedding_models.gnn.formula_net.formula_net import FormulaNetEdges
from models.end_to_end.tactic_models.tac_models import TacModel, TacWrapper
from models.end_to_end.tactic_models.tacticzero.model import TacticZeroTacModel
from models.get_model import get_model

warnings.filterwarnings('ignore')


def get_model_dict(prefix, state_dict):
    return {k[len(prefix) + 1:]: v for k, v in state_dict.items()
            if k.startswith(prefix)}


def load_pretrained_encoders(self, encoder_premise, encoder_goal):
    ckpt_dir = self.config.pretrain_ckpt
    ckpt = torch.load(ckpt_dir)['state_dict']
    encoder_premise.load_state_dict(get_model_dict('embedding_model_premise', ckpt))
    encoder_goal.load_state_dict(get_model_dict('embedding_model_goal', ckpt))


class HOL4TacGen(TacModel):
    def __init__(self, config):
        super().__init__()
        pretrain = config.pretrain

        # default policy models
        context_net = ContextPolicy()
        tac_net = TacPolicy(len(config.tac_config.tactic_pool))
        arg_net = ArgPolicy(len(config.tac_config.tactic_pool), config.model_config.model_attributes.embedding_dim)
        term_net = TermPolicy(len(config.tac_config.tactic_pool), config.model_config.model_attributes.embedding_dim)

        induct_net = FormulaNetEdges(config.model_config.model_attributes.vocab_size,
                                     config.model_config.model_attributes.embedding_dim,
                                     num_iterations=3, global_pool=False,
                                     batch_norm=False)

        encoder_premise = get_model(config.model_config)
        encoder_goal = get_model(config.model_config)

        if pretrain:
            logger.info("Loading pretrained encoder models..")
            load_pretrained_encoders(encoder_premise, encoder_goal)

        tac_model = TacticZeroTacModel(goal_net=context_net,
                                       tac_net=tac_net,
                                       arg_net=arg_net,
                                       term_net=term_net,
                                       induct_net=induct_net,
                                       encoder_premise=encoder_premise,
                                       encoder_goal=encoder_goal,
                                       config=config
                                       )

        if config.distributed:
            self.tac_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(
                TacWrapper).remote(
                tac_model=tac_model)
        else:
            self.tac_model = tac_model

    def get_tactics(self, goal, premises):
        if self.distributed:
            tactics = ray.get(self.tac_model.get_tactics.remote(goal, premises))
        else:
            tactics = self.tac_model.get_tactics(goal.goal, premises)
        return tactics
