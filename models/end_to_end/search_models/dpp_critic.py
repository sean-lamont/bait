from __future__ import division, absolute_import, print_function

import math

import ray

from experiments.end_to_end.proof_node import InternalNode
from models.end_to_end.search_models.search_models import Search

from dppy.finite_dpps import FiniteDPP


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

    def reset(self, root):
        self.__init__(self.critic_model)
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
        valid_states = [(goal,score) for goal, (score, embedding) in self.state_data.items() if score > -math.inf]

        if not valid_states:
            return None


        valid_states = sorted(valid_states, key = lambda x: x[1], reverse=True)[:self.max_candidates]

        # todo could optimise? E.g. preallocate np array?
        
        # todo join np matrix
        emb_matrix = [self.state_data[x[0]][1] * self.state_data[x[0]][0] for x in valid_states]


        # likelihood matrix for DPP
        emb_matrix = emb_matrix @ emb_matrix.T


        try:
            DPP = FiniteDPP('likelihood', **{'L': emb_matrix})

            # rng = np.random.RandomState(1)
            DPP.sample_exact_k_dpp(size=num_filtered, mode='KuTa12')  # ,rng=rng)
            chosen_inds = DPP.list_of_samples[0]

        except Exception as e:
            logger.error(f"Error sampling from DPP: {e}")
            # take the top num_filtered tactics if DPP fails
            chosen_inds = [[i for i in range(self.num_filtered)]][0]


        chosen_goals = [valid_states[i][0] for i in chosen_inds]

        # Only allow one exploration of each node
        [self.state_data[goal] = (-math.inf, None) for goal in chosen_goals]

        return [self.nodes[goal] for goal in chosen_goals]


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
