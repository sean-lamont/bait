from __future__ import division, absolute_import, print_function

import math
import traceback

import ray

from experiments.end_to_end.proof_node import InternalNode
from models.end_to_end.search_models.search_models import Search

from dppy.finite_dpps import FiniteDPP
import numpy as np
from loguru import logger

from numpy.linalg import inv


# implements eqn 42 from https://arxiv.org/pdf/1207.6083, assuming first index is alwasys the max score
def sample_conditional_dpp(L,k=2):
    '''
    Wrapper function for the sample_dpp Matlab code written by Alex Kulesza
    Given a kernel matrix L, returns a sample from a k-DPP.
    The code is hacked in a way that if a set A is provied, samples from a conditional
    dpp given A are produced
    L:     kernel matrix
    set:   index of the conditional elements. Integer numpy array containing the locations
            (starting in zero) relative to the rows of L.
    k:     size of the sample from the DPP
    '''
    set0 = np.array([0]) # matlab starts counting in one
    # Calculate the kernel for the marginal
    Id = np.array([1]*L.shape[0])
    Id[set0] = 0
    Id = np.diag(Id)
    L_compset_full = inv(Id + L)
    L_minor = inv(np.delete(np.delete(L_compset_full,tuple(set0), axis=1),tuple(set0),axis=0))
    L_compset = L_minor - np.diag([1]*L_minor.shape[0])

    DPP = FiniteDPP('likelihood', **{'L': L_compset})

    # inds = DPP.sample_exact()
    sample = DPP.sample_exact_k_dpp(size=k-1, mode='KuTa12')  # ,rng=rng)

    # Compute the sample
    return [0] + [s + 1 for s in sample] # add one as conditional DPP removed first element

class DPPCritic(Search):
    """

    Runs DPP over proof state embeddings, with scores determined by a critic model. 

    """

    def __init__(self, critic_model, state_encoder, num_filtered, max_candidates):
        super().__init__()

        # neural model to provide a score for the goals
        self.critic_model = critic_model

        # model to generate encodings for goal states (expected to return unit norm embeddings)
        self.state_encoder = state_encoder

        # map goal to score and embeddings for models
        self.state_data = {}

        # number of goals chosen per search iteration
        self.num_filtered = num_filtered

        # maximum number of states to consider in DPP (if None, select all) 
        self.max_candidates = max_candidates

        self.explored = set()

    def reset(self, root):
        self.__init__(self.critic_model, self.state_encoder, self.num_filtered, self.max_candidates)
        self.root = root

        if isinstance(root, InternalNode):
            self.nodes[root.goal] = root

            # Initialise scores for root
            score = ray.get(self.critic_model.run.remote([self.root.goal]))

            embedding = ray.get(self.state_encoder.encode.remote([self.root.goal]))

            self.state_data[self.root.goal] = (score[0], embedding[0].cpu().numpy())

    def get_goals(self):

        # create DPP matrix from currently unexplored states, scaling each by score

        # only take the top max_candidates valid states
        # valid_states = [(goal, score) for goal, (score, embedding) in self.state_data.items() if score > -math.inf]

        valid_states = [(goal, score) for goal, (score, embedding) in self.state_data.items() if goal not in self.explored and self.nodes[goal].is_explored == False]

        if not valid_states:
            return None

        if self.max_candidates is not None:
            valid_states = sorted(valid_states, key=lambda x: x[1], reverse=True)[:self.max_candidates]
        else:
            valid_states = sorted(valid_states, key=lambda x: x[1], reverse=True)


        if len(valid_states) <= self.num_filtered:
            # chosen_inds = [i for i in range(len(valid_states))]
            chosen_inds = [0]
        else:
            try:
                # normalise scores to be in [0,1], plus an epsilon (1) to avoid DPP errors
                max_score = max(valid_states, key=lambda x: x[1])[1]
                min_score = min(valid_states, key=lambda x: x[1])[1]
                valid_states = [(s[0], (s[1] - min_score + 1) / (max_score - min_score)) for s in valid_states]

                emb_matrix = np.array([self.state_data[goal][1] * score for goal, score in valid_states])


                # likelihood matrix for DPP
                emb_matrix = emb_matrix @ emb_matrix.T

                # sample under conditional DPP, where max score is always chosen (idx always 0 as sorted)
                chosen_inds = sample_conditional_dpp(emb_matrix, k=self.num_filtered)

                # DPP = FiniteDPP('likelihood', **{'L': emb_matrix})
                # # rng = np.random.RandomState(1)
                # DPP.sample_exact_k_dpp(size=self.num_filtered, mode='KuTa12')  # ,rng=rng)
                # chosen_inds = DPP.list_of_samples[0]
                # print ('DPP Success')

            except Exception as e:
                # logger.error(f"Error sampling from DPP: {e}")
                # traceback.print_exc()
                # take the top num_filtered tactics if DPP fails
                # chosen_inds = [i for i in range(self.num_filtered)]
                chosen_inds = [0]

        # print(valid_states, len(valid_states), chosen_inds)
        chosen_goals = [valid_states[i][0] for i in chosen_inds]

        ret = []
        for goal in chosen_goals:
            ret.append((self.nodes[goal], self.state_data[goal][0]))
            # Only allow one exploration of each node
            # self.state_data[goal] = (-math.inf, None)
            self.explored.add(goal)

        return ret

    def process_responses(self, responses):
        for response in responses:
            result = response.dst

            for result_node in result:
                # Don't search proved/explored/queued nodes
                if isinstance(result_node, InternalNode) and result_node.goal not in self.nodes:
                    self.nodes[result_node.goal] = result_node

                    # Initialise scores for root
                    score = ray.get(self.critic_model.run.remote([result_node.goal]))

                    embedding = ray.get(self.state_encoder.encode.remote([result_node.goal]))

                    self.state_data[result_node.goal] = (score[0], embedding[0].cpu().numpy())

        return
