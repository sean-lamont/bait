import pickle
from typing import Dict, Any

import einops
import lightning.pytorch as pl
import torch.nn.functional as F
import torch.optim
from torch.distributions import Categorical
from torch_geometric.data import Batch

from data.HOL4.utils import ast_def
from data.utils.graph_data_utils import transform_expr, transform_batch
from environments.HOL4.tacticzero_old.env_wrapper import *

torch.set_float32_matmul_precision("medium")

"""

Implementation of TacticZero in our End-to-End AI-ITP framework. 

Differs from the original approach:
    - Offline training
    - Not restricted to a single search strategy
    - Doesn't "replay" proofs


Online/replays could be implemented by re-training the model after each proof attempt with the most recent trace
for online, and most recently proved trace for replays.

Fringes as in the original approach are delegated to Search. Original Fringe training can be implemented by 
using the Fringe search model, by processing traces to give the probability and reward based on the final outcome,
and by training on these with policy gradient.

---- Currently only implemented and tested for End-to-End evaluation. Training still needs to be written, 
along with DataModule.  ---- 

"""


class TacticZeroTacModel(pl.LightningModule):
    def __init__(self,
                 goal_net,
                 tac_net,
                 arg_net,
                 term_net,
                 induct_net,
                 encoder_premise,
                 encoder_goal,
                 config) -> None:

        super().__init__()

        self.save_hyperparameters()

        self.goal_net = goal_net
        self.tac_net = tac_net
        self.arg_net = arg_net
        self.term_net = term_net
        self.induct_net = induct_net
        self.encoder_premise = encoder_premise
        self.encoder_goal = encoder_goal

        self.thms_tactic = list(config.tac_config.thms_tactic)
        self.thm_tactic = list(config.tac_config.thm_tactic)
        self.term_tactic = list(config.tac_config.term_tactic)
        self.no_arg_tactic = list(config.tac_config.no_arg_tactic)
        self.tactic_pool = list(config.tac_config.tactic_pool)

        self.config = config

        self.expr_dict = {}

        self.vocab = pickle.load(open('data/HOL4/data/vocab.pk', 'rb'))

        # number of candidate tactics generated per goal
        self.num_val_samples = config.num_val_samples if hasattr(config, 'num_val_samples') else 0

    # todo load model

    def converter(self, data_list):
        if self.config.data_config.data_type == 'fixed':
            return [d.strip().split() for d in data_list]

        # for now, add every unseen expression to database
        for d in data_list:
            if d not in self.expr_dict:
                self.expr_dict[d] = transform_expr(expr=ast_def.goal_to_dict(d),
                                                   data_type=self.config.data_config.data_type,
                                                   vocab=self.vocab,
                                                   config=self.config)

        batch = [self.expr_dict[d] for d in data_list]

        return transform_batch(batch, self.config.data_config)

    # todo
    ############
    # Training #
    ############

    def forward(self, goal):
        pass

    def training_step(self, batch, batch_idx: int):
        pass

    ##############
    # Validation #
    ##############

    def validation_step(self, batch: Dict[str, Any], _) -> None:
        pass

    def run_eval(self) -> None:
        pass

    ##############
    # Evaluation #
    ##############

    def get_tac(self, tac_input):
        tac_probs = self.tac_net(tac_input)
        tac_m = Categorical(tac_probs)
        tac = tac_m.sample()
        tac_prob = tac_m.log_prob(tac)
        tac_tensor = tac.to(self.device)
        return tac_tensor, tac_prob

    # determine term for induction based on data type (graph, fixed, sequence)
    def get_term_tac(self, target_goal, target_representation, tac):
        arg_probs = []

        induct_expr = self.converter([target_goal])

        if self.config.data_config.type == 'graph':
            induct_expr = Batch.to_data_list(induct_expr)

            assert len(induct_expr) == 1
            induct_expr = induct_expr[0]

            labels = ast_def.goal_to_dict(target_goal)['labels']
            induct_expr = induct_expr.to(self.device)
            induct_expr.labels = labels
            tokens = [[t] for t in induct_expr.labels if t[0] == "V"]
            token_inds = [i for i, t in enumerate(induct_expr.labels) if t[0] == "V"]
            if tokens:
                # Encode all nodes in graph
                induct_nodes = self.induct_net(induct_expr)
                # select representations of Variable nodes with ('V' label only)
                token_representations = torch.index_select(induct_nodes, 0, torch.tensor(token_inds).to(self.device))
                target_representations = einops.repeat(target_representation, '1 d -> n d', n=len(tokens))

        else:
            tokens = target_goal.split()
            tokens = list(dict.fromkeys(tokens))
            if self.config.data_config.type == 'sequence':
                tokens = [t for t in tokens if t[0] == "V"]
                tokens_ = self.converter(tokens)
                tokens_ = (tokens_[0].to(self.device), tokens_[1].to(self.device))
                if tokens_:
                    token_representations = self.encoder_goal(tokens_).to(self.device)
                    target_representations = einops.repeat(target_representation, '1 d -> n d',
                                                           n=token_representations.shape[0])
                    tokens = [[t] for t in tokens]

            elif self.config.data_config.type == 'fixed':
                tokens = [[t] for t in tokens if t[0] == "V"]
                if tokens:
                    token_representations = self.encoder_goal(tokens).to(self.device)
                    target_representations = einops.repeat(target_representation, '1 d -> n d', n=len(tokens))
            else:
                raise NotImplementedError("Induction for non-supported data type")

        if tokens:
            # pass through term_net
            candidates = torch.cat([token_representations, target_representations], dim=1)
            scores = self.term_net(candidates, tac)
            term_probs = F.softmax(scores, dim=0)
            term_m = Categorical(term_probs.squeeze(1))

            term = term_m.sample()

            arg_probs.append(term_m.log_prob(term))
            tm = tokens[term][0][1:]  # remove headers, e.g., "V" / "C" / ...
            tactic = "Induct_on `{}`".format(tm)

        else:
            arg_probs.append(torch.tensor(0))
            tactic = "Induct_on"

        return tactic, arg_probs

    def get_arg_tac(self, target_representation,
                    encoded_fact_pool,
                    tac,
                    candidate_args,
                    ):

        hidden0 = hidden1 = target_representation
        hidden0 = hidden0.to(self.device)
        hidden1 = hidden1.to(self.device)

        hidden = (hidden0, hidden1)

        # concatenate the candidates with hidden states.
        hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
        hiddenl = [hc.unsqueeze(0) for _ in range(len(candidate_args))]
        hiddenl = torch.cat(hiddenl)

        candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
        candidates = candidates.to(self.device)
        input = tac

        # run it once before predicting the first argument
        hidden, _ = self.arg_net(input, candidates, hidden)

        # the indices of chosen args
        arg_step = []
        arg_step_probs = []

        if self.tactic_pool[tac] in self.thm_tactic:
            arg_len = 1
        else:
            arg_len = self.config.arg_len

        for i in range(arg_len):
            hidden, scores = self.arg_net(input, candidates, hidden)
            arg_probs = F.softmax(scores, dim=0)
            arg_m = Categorical(arg_probs.squeeze(1))

            arg = arg_m.sample()

            arg_step.append(arg)
            arg_step_probs.append(arg_m.log_prob(arg))

            hidden0 = hidden[0].squeeze().repeat(1, 1, 1)
            hidden1 = hidden[1].squeeze().repeat(1, 1, 1)

            # encoded chosen argument
            input = encoded_fact_pool[arg].unsqueeze(0)

            # renew candidates
            hc = torch.cat([hidden0.squeeze(), hidden1.squeeze()])
            hiddenl = [hc.unsqueeze(0) for _ in range(len(candidate_args))]
            hiddenl = torch.cat(hiddenl)

            # appends both hidden and cell states
            candidates = torch.cat([encoded_fact_pool, hiddenl], dim=1)
            candidates = candidates.to(self.device)

        tac = tactic_pool[tac]
        arg = [candidate_args[j] for j in arg_step]

        return "<arg>".join([tac] + arg), arg_step_probs

    def get_tactics(self, goals, premises):
        target_goal = goals

        target_representation = self.encoder_goal([target_goal]).to(self.device)

        encoded_fact_pool = self.encoder_premise(premises).to(self.device)

        actions = []
        for _ in range(self.config.num_sampled_tactics):
            tac, tac_prob = self.get_tac(target_representation)

            if self.tactic_pool[tac] in self.no_arg_tactic:
                tactic = self.tactic_pool[tac]
                arg_probs = [torch.tensor(1)]

            elif self.tactic_pool[tac] == "Induct_on":
                tactic, arg_probs = self.get_term_tac(target_goal=target_goal,
                                                      target_representation=target_representation,
                                                      tac=tac)
            else:
                tactic, arg_probs = self.get_arg_tac(target_representation=target_representation,
                                                     encoded_fact_pool=encoded_fact_pool,
                                                     tac=tac,
                                                     candidate_args=premises,
                                                     )

            # todo if we want to train as done originally (or to importance sample)
            #  need to keep track of tac and arg prob separately
            # could do e.g. tactic trace, then process that as part of prepare_data

            # combine probabilities
            arg_prob = torch.prod(torch.stack(arg_probs))
            action_prob = tac_prob * arg_prob

            action = tactic, action_prob.item()
            actions.append(action)

        return actions
