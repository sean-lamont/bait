from __future__ import division, absolute_import, print_function

import math
import random

import ray
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from experiments.end_to_end.proof_node import InternalNode, Status
from models.end_to_end.search_models.search_models import Search


class FringeSearch(Search):
    """

    Fringe approach from TacticZero.
    Score each goal in a fringe and take the product, then take the first goal.

    Similar to BestFS without subgoal separation, except scored by goal model rather than tactic logprob
    At most one new fringe generated per tactic application. Hence UpDown is superior in that it finds all possible fringes

    """

    def __init__(self, goal_model):
        super().__init__()

        # list of fringes
        self.fringes = []

        # single fringe, i.e. list of goals to prove which will satisfy the original goal
        self.last_fringe = None
        self.last_chosen = None

        # neural model to provide a score for the goals
        self.goal_model = goal_model

        # map goal to score from model
        self.scores = {}

    def reset(self, root):
        self.__init__(self.goal_model)
        self.root = root

        if isinstance(root, InternalNode):
            self.nodes[root.goal] = root

            # Initialise scores for root
            scores = ray.get(self.goal_model.run.remote([self.root.goal]))

            self.scores[self.root.goal] = scores[0]

            self.fringes = [{self.root.goal}]

            self.last_fringe = self.fringes[0]

            self.last_chosen = set()

    def get_goals(self):
        fringe_scores = []

        # go through fringes and remove any goals which are already proven
        # and update fringe scores
        for i, fringe in enumerate(self.fringes):
            fringe = {goal for goal in fringe if self.nodes[goal].status != Status.PROVED}

            fringe_score = torch.FloatTensor(0.0)
            for goal in fringe:

                if self.nodes[goal].is_explored:
                    score = -math.inf
                else:
                    if goal not in self.scores:
                        # Initialise scores for root
                        self.scores[goal] = ray.get(self.goal_model.run.remote([goal]))[0]

                    score = self.scores[goal]

                fringe_score += score

            self.fringes[i] = fringe

            fringe_scores.append(fringe_score)

        fringe_scores = torch.stack(fringe_scores)

        fringe_probs = F.softmax(fringe_scores, dim=0)
        fringe_m = Categorical(fringe_probs)

        fringe = fringe_m.sample()

        fringe_prob = fringe_m.log_prob(fringe)

        chosen_fringe = self.fringes[fringe]

        chosen_goal = self.nodes[random.choice(tuple(chosen_fringe))]

        if chosen_goal.is_explored:
            return None
        else:
            self.last_fringe = chosen_fringe

            self.last_chosen = chosen_goal

            return [(chosen_goal, fringe_prob.item())]

    def process_responses(self, responses):
        # ensure only the expected goal was worked on
        assert all([response.src.goal == self.last_chosen for response in responses])

        # previous fringe excluding goal that was worked on
        prev_fringe = self.last_fringe - self.last_chosen

        new_fringes = []
        for response in responses:
            result = response.dst

            # invalid/error response gives no new fringe
            if any([result_node.status == Status.FAILED for result_node in result]):
                continue

            new_goals = set()

            for result_node in result:
                # Don't search proved/explored/queued nodes
                if isinstance(result_node, InternalNode):
                    self.nodes[result_node.goal] = result_node
                    new_goals.add(result_node.goal)

            new_fringe = prev_fringe | new_goals

            if new_fringe not in new_fringes:
                new_fringes.append(new_fringe)

        self.fringes.extend(new_fringes)

        return
