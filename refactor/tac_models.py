
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from loguru import logger

from refactor.proof_node import *


class TacModel:
    @abstractmethod
    def get_tactics(self, goals, env):
        return


# todo make tac_gen and retriever more system agnostic
class ReProverTacGen(TacModel):
    def __init__(self, tac_model, num_sampled_tactics=64):
        super().__init__()
        self.tac_model = tac_model
        self.num_sampled_tactics = num_sampled_tactics

    def get_tactics(self, ts, premises):
        path, theorem, position = premises

        tactics = self.tac_model.generate(
            state=ts,
            file_path=path,
            theorem_full_name=theorem.full_name,
            theorem_pos=position,
            num_samples=self.num_sampled_tactics,
        )

        logger.debug(f"Tactic suggestions: {tactics}")
        return tactics


# DPO loss from paper,

# import torch.nn.functional as F
# def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
#     """
#     pi_logps: policy logprobs, shape (B,)
#     ref_logps: reference model logprobs, shape (B,)
#     yw_idxs: preferred completion indices in [0, B-1], shape (T,)
#     yl_idxs: dispreferred completion indices in [0, B-1], shape (T,)
#     beta: temperature controlling strength of KL penalty
#     Each pair of (yw_idxs[i], yl_idxs[i]) represents the
#     indices of a single preference pair.
#     """
#     pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs]
#     ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]
#     pi_logratios = pi_yw_logps - pi_yl_logps
#     ref_logratios = ref_yw_logps - ref_yl_logps
#     losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
#     rewards = beta * (pi_logps - ref_logps).detach()
#     return losses, rewards
