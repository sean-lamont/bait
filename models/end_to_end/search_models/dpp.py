from __future__ import division, absolute_import, print_function

from typing import List

import ray
from loguru import logger
from experiments.end_to_end.proof_node import InternalNode, Edge
from models.end_to_end.search_models.search_models import Search


class DPPSearch(Search):
    def __init__(self, diversity_model, num_filtered, temperature=1., scale=1.):
        super().__init__()
        self.priority_queue = []

        # record the chosen nodes for further analysis
        self.search_trace = []

        self.diversity_model = diversity_model
        self.num_filtered = num_filtered
        self.temperature = temperature
        self.scale = scale

    def reset(self, root):
        self.__init__(self.diversity_model, self.num_filtered, self.temperature, self.scale)
        self.root = root
        if isinstance(root, InternalNode):
            self.priority_queue = [root]
            self.nodes[root.goal] = root
            self.theorem = root.data['theorem']

    def get_goals(self):
        self.priority_queue = sorted(self.priority_queue, key=lambda x: x.cumulative_logprob)
        if len(self.priority_queue) > 0:
            search_node = self.priority_queue.pop()
            # if node was set to explored since being added (e.g. if ancestor was proven)
            if search_node.is_explored:
                return self.get_goals()

            return [(search_node, search_node.cumulative_logprob)]
        else:
            return None

    # assumes only one node expanded at a time
    def process_responses(self, responses: List[Edge]):
        assert all([response.src == responses[0].src for response in responses])

        tactics = [response.tactic for response in responses]

        goal = responses[0].src

        goal.data['original_tacs'] = tactics

        state = goal.data['augmented_state'] if hasattr(goal, 'data') and 'augmented_state' in goal.data else goal.goal

        # get valid tactics:
        valid_tactics = set()
        for response in responses:
            result = response.dst

            for result_node in result:
                # Don't search proved/explored/queued nodes
                if isinstance(result_node,
                              InternalNode) and result_node not in self.priority_queue and not result_node.is_explored:
                    self.nodes[result_node.goal] = result_node
                    tac = (response.tactic, response.tac_logprob)
                    if tac not in valid_tactics:
                        valid_tactics.add(tac)

        valid_tactics = list(valid_tactics)

        if valid_tactics:
            if len(valid_tactics) <= self.num_filtered:
                tactics = {valid_tactics[i][0] for i in range(len(valid_tactics))}
            else:
                # filter with diversity model
                inds = ray.get(
                    self.diversity_model.filter_tacs.remote(valid_tactics,
                                                            self.num_filtered,
                                                            state=state, theorem=self.theorem,
                                                            temperature=self.temperature, scale=self.scale))

                tactics = {valid_tactics[i][0] for i in sorted(inds[0])}

            responses_ = [response for response in responses if response.tactic in tactics]

            for response in responses_:
                result = response.dst

                for result_node in result:
                    # Don't search proved/explored/queued nodes
                    if isinstance(result_node,
                                  InternalNode) and result_node not in self.priority_queue and not result_node.is_explored:
                        self.priority_queue.append(result_node)

        self.search_trace.append(responses)

        return
