from __future__ import division, absolute_import, print_function

import copy
import math
from typing import List

import ray
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from experiments.end_to_end.proof_node import InternalNode, Status, Edge
from models.end_to_end.search_models.search_models import Search, GoalModel


class UpDown(Search):
    def __init__(self, goal_model: GoalModel):
        super().__init__()
        self.goal_model = goal_model
        self.initial_scores = {}
        self.updated_scores = {}
        self.search_trace = []

    def reset(self, root):
        self.__init__(self.goal_model)
        self.root = root
        self.search_trace = []

        if isinstance(root, InternalNode):
            self.nodes[root.goal] = root

            # Initialise scores for root
            scores = ray.get(self.goal_model.run.remote([self.root.goal]))

            self.initial_scores[root.goal] = scores[0].item()
            self.updated_scores[root.goal] = scores[0].item()

    # sampling version
    def get_goals(self):
        fringe_scores = []

        node_scores = {}

        for goal, node in self.nodes.items():
            if node.is_explored:
                continue
            # Take the score for a node as the probability of proving that goal,
            # multiplied by the probability of proving the best context of that goal
            # (i.e how likely to prove the original goal, assuming this goal is used)
            if node.context and len(node.context[0]) > 0:
                score = self.initial_scores[goal] + max(
                    [sum([self.updated_scores[ctx] for ctx in context]) for context in node.context])

            else:
                score = self.initial_scores[goal]

            node_scores[node.goal] = score

            fringe_scores.append(score)

        # sample from fringe scores
        fringe_probs = F.softmax(torch.FloatTensor(fringe_scores), dim=0)
        fringe_m = Categorical(fringe_probs)

        sampled_ind = fringe_m.sample()

        sampled_score = fringe_scores[sampled_ind]

        # find fringe for selected node by choosing all goals with the same score.
        # (may include other goals with same score not in fringe)
        chosen_fringe = []

        for goal, score in node_scores.items():
            if score == sampled_score:
                chosen_fringe.append((self.nodes[goal], sampled_score))

        self.search_trace.append(
            copy.deepcopy(([f[0].goal for f in chosen_fringe], node_scores, self.initial_scores, self.updated_scores)))

        return chosen_fringe

    # greedy version
    # def get_goals(self):
    #     best_score = -math.inf
    #     best_node = None
    #
    #     node_scores = {}
    #     for goal, node in self.nodes.items():
    #         if node.is_explored:
    #             continue
    #         # Take the score for a node as the probability of proving that goal,
    #         # multiplied by the probability of proving the best context of that goal
    #         # (i.e how likely to prove the original goal, assuming this goal is used)
    #         if node.context and len(node.context[0]) > 0:
    #             score = self.initial_scores[goal] + max(
    #                 [sum([self.updated_scores[ctx] for ctx in context]) for context in node.context])
    #
    #         else:
    #             score = self.initial_scores[goal]
    #
    #         node_scores[node.goal] = score
    #         if score > best_score:
    #             best_score = score
    #             best_node = node
    #
    #     if not best_node:
    #         return []
    #
    #     # find fringe for selected node by choosing all goals with the same score.
    #     # (may include other goals with same score not in fringe)
    #     best_fringe = []
    #
    #     for goal, score in node_scores.items():
    #         if score == best_score:
    #             best_fringe.append((self.nodes[goal], best_score))
    #
    #     self.search_trace.append(
    #         copy.deepcopy(([f[0].goal for f in best_fringe], node_scores, self.initial_scores, self.updated_scores)))
    #
    #     return best_fringe

    def _up_step(self, node):
        if node.out_edges:
            if node.status == Status.PROVED:
                best_score = 0
            else:
                best_score = -math.inf
                valid_edges = [edge for edge in node.out_edges if all([isinstance(d, InternalNode) for d in edge.dst])]
                for edge in valid_edges:
                    edge_score = 0
                    for sib in edge.dst:
                        edge_score += self.updated_scores[sib.goal]

                    if edge_score > best_score:
                        best_score = edge_score

            if node.visit_count >= node.max_expansions:
                self.initial_scores[node.goal] = -math.inf
                node.is_explored = True

            up_score = max(self.initial_scores[node.goal], best_score)

            # todo scale breadth as explored?
            if up_score != self.updated_scores[node.goal]:
                self.updated_scores[node.goal] = up_score
                parents = set([edge.src for edge in node.in_edges])
                for parent in parents:
                    self._up_step(parent)

    def process_responses(self, responses: List[Edge]):
        for response in responses:
            result = response.dst

            # find new nodes from response, and compute their provable score
            new_nodes = []
            for result_node in result:
                if isinstance(result_node, InternalNode):
                    if result_node.goal not in self.nodes:
                        new_nodes.append(result_node)
                        self.nodes[result_node.goal] = result_node

            if new_nodes:
                scores = ray.get(self.goal_model.run.remote([g.goal for g in new_nodes]))

                # Initialise provable_score/up_score for new internal nodes
                for i, node_ in enumerate(new_nodes):
                    scaled_score = (scores[i] + (node_.depth * math.log(0.99))).item()
                    self.initial_scores[node_.goal] = scaled_score
                    self.updated_scores[node_.goal] = scaled_score
                    assert self.nodes[node_.goal] is node_

        to_update = set([response.src for response in responses])
        for search_node in to_update:
            self._up_step(search_node)

        self.search_trace[-1] = (self.search_trace[-1], responses)

        return
