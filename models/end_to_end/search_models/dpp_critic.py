from __future__ import division, absolute_import, print_function

import math

import ray

from experiments.end_to_end.proof_node import InternalNode
from models.end_to_end.search_models.search_models import Search


class DPPCritic(Search):
    """

    Runs DPP over proof state embeddings, with scores determined by a critic model. 

    """

    def __init__(self, critic_model, state_encoder):
        super().__init__()

        # neural model to provide a score for the goals
        self.critic_model = critic_model

        # model to generate encodings for goal states (expected to return unit norm embeddings)
        self.state_encoder = state_encoder

        # map goal to score and embeddings for models
        self.state_data = {}

    def reset(self, root):
        self.__init__(self.critic_model)
        self.root = root

        if isinstance(root, InternalNode):
            self.nodes[root.goal] = root

            # Initialise scores for root
            score = ray.get(self.critic_model.run.remote([self.root.goal]))

            embedding = ray.get(self.state_encoder.encode.remote([self.root.goal]))

            self.state_data[self.root.goal] = (state_data[0], embedding[0].cpu().numpy())

    def get_goals(self):

        # create DPP matrix from currently unexplored states, scaling each by score



        max_goal = max(self.state_data, key=self.state_data.get)
        chosen_node = self.nodes[max_goal]

        if chosen_node.is_explored:
            return None
        else:
            # Only allow one exploration of each node
            self.state_data[max_goal] = -math.inf
            return [(chosen_node, self.state_data[max_goal])]

    def process_responses(self, responses):
        for response in responses:
            result = response.dst

            for result_node in result:
                # Don't search proved/explored/queued nodes
                if isinstance(result_node, InternalNode) and result_node.goal not in self.nodes:
                    self.nodes[result_node.goal] = result_node
                    self.state_data[result_node.goal] = ray.get(self.critic_model.run.remote([result_node.goal]))[0]

        return
