from __future__ import division, absolute_import, print_function

import copy
import math

import ray

from experiments.end_to_end.proof_node import InternalNode
from models.end_to_end.search_models.search_models import Search


class CriticGuidedSearch(Search):
    """

    Critic Guided Search from InternLM2.5. Assumes no subgoal separation, uses a critic model to score
    states and takes the highest state at each time step.

    """

    def __init__(self, goal_model):
        super().__init__()

        # neural model to provide a score for the goals
        self.goal_model = goal_model

        # map goal to score from model
        self.scores = {}

        self.explored = set()

    def reset(self, root):
        self.__init__(self.goal_model)
        self.root = root

        if isinstance(root, InternalNode):
            self.nodes[root.goal] = root

            # Initialise scores for root
            scores = ray.get(self.goal_model.run.remote([self.root.goal]))

            self.scores[self.root.goal] = scores[0]


    def get_goals(self):
        valid_goals = [(goal, score) for goal,score in self.scores.items() if goal not in self.explored and self.nodes[goal].is_explored == False]

        if not valid_goals:
            return None

        max_goal = max(valid_goals, key=lambda x: x[1])[0]
        chosen_node = self.nodes[max_goal]

        assert not chosen_node.is_explored

        self.explored.add(max_goal)

        return [(chosen_node, self.scores[max_goal])]

    def process_responses(self, responses):
        for response in responses:
            result = response.dst

            for result_node in result:
                # Don't search proved/explored/queued nodes
                if isinstance(result_node, InternalNode) and result_node.goal not in self.nodes:
                    self.nodes[result_node.goal] = result_node
                    self.scores[result_node.goal] = ray.get(self.goal_model.run.remote([result_node.goal]))[0]
        return
