from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytorch_lightning as pl

import ray

from refactor.proof_node import *


class GoalModel:
    def __init__(self, model):
        self.model = model

    def run(self, goals):
        scores = self.model.batch_generate(goals)
        return scores


# Trains goal model based on binary proven/unproven loss
class BinaryGoalModel(pl.LightningModule):
    pass


# Trains goal model based on proof length objective based on Polu et al.
class ProofLengthGoalModel(pl.LightningModule):
    pass


class Search:
    def __init__(self):
        self.nodes = {}
        self.root = None

    @abstractmethod
    def reset(self, root):
        return

    @abstractmethod
    def get_goals(self):
        return

    @abstractmethod
    def process_response(self, response: Edge):
        return


class UpDown(Search):
    def __init__(self, goal_model: GoalModel):
        super().__init__()
        self.goal_model = goal_model

    def reset(self, root):
        self.__init__(self.goal_model)
        self.root = root
        self.nodes[root.goal] = root

        # todo move to model
        node_goals = ['<extra_id_0>' + self.root.goal]
        # Initialise scores for root
        scores = ray.get(self.goal_model.run.remote(node_goals))

        self.root.provable_score = scores[0]
        self.root.up_score = scores[0]

    # todo sample?
    # todo return goal plus the context (i.e. best fringe)?
    # todo take as parameter # of goals?
    def get_goals(self):
        best_score = -math.inf
        best_node = None

        for goal, node in self.nodes.items():
            if node.is_explored:
                continue
            # Take the score for a node as the probability of proving that goal,
            # multiplied by the probability of proving the best context of that goal
            # (i.e how likely to prove the original goal, assuming this goal is used)
            if node.context and len(node.context[0]) > 0:
                score = node.provable_score + max(
                    [sum([self.nodes[ctx].up_score for ctx in context]) for context in node.context])
            else:
                score = node.provable_score

            if score > best_score:
                best_score = score
                best_node = node

        # todo find fringe for selected node by choosing all goals with the same score.

        if best_node:
            return [(best_node, best_score)]
        else:
            return []

    def _up_step(self, node):
        if node.out_edges:
            best_score = -math.inf
            for edge in node.out_edges:
                edge_score = 0
                for sib in edge.dst:
                    edge_score += sib.up_score

                if edge_score > best_score:
                    best_score = edge_score

            if node.visit_count >= node.max_expansions:
                node.provable_score = -math.inf
                node.is_explored = True

            up_score = max(node.provable_score, best_score)

            # todo scale breadth as it's explored?
            if up_score != node.up_score:
                node.up_score = up_score
                parents = set([edge.src for edge in node.in_edges])
                for parent in parents:
                    self._up_step(parent)

    # Assume response is a single edge in this case
    def process_response(self, response: Edge):
        search_node = response.src
        result = response.dst

        # find new nodes from response, and compute their provable score
        new_nodes = []
        for result_node in result:
            if isinstance(result_node, InternalNode):
                if result_node.goal not in self.nodes:
                    new_nodes.append(result_node)
                    self.nodes[result_node.goal] = result_node

        if new_nodes:
            # todo move to model
            node_goals = ['<extra_id_0>' + node_.goal for node_ in new_nodes]

            scores = ray.get(self.goal_model.run.remote(node_goals))

            # Initialise provable_score/up_score for new internal nodes
            for i, node_ in enumerate(new_nodes):
                node_.provable_score = (scores[i] + (node_.depth * math.log(0.95))).item()
                node_.up_score = node_.provable_score
                assert self.nodes[node_.goal] is node_

        self._up_step(search_node)

        return


# based on cumulative logprob, maintain priority queue, pop for get_goals, populate in process_response
# currently only permits goals to be expanded once
class BestFS(Search):
    def __init__(self):
        super().__init__()
        self.priority_queue = []

    def reset(self, root):
        self.__init__()
        self.root = root
        self.priority_queue = [root]
        self.nodes[root.goal] = root

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

    def process_response(self, response: Edge):
        result = response.dst

        for result_node in result:
            # Don't search proved/explored/queued nodes
            if isinstance(result_node,
                          InternalNode) and result_node not in self.priority_queue and not result_node.is_explored:
                self.nodes[result_node.goal] = result_node
                self.priority_queue.append(result_node)

        return


# Similar to BestFS, but score each goal in a state and take the product, then take the first goal
class FringeSearch(Search):
    def get_goals(self):
        pass

    def process_response(self, response):
        pass

    def reset(self, root):
        pass


class BFS(Search):
    def reset(self, root):
        pass

    def get_goals(self):
        pass

    def process_response(self, response):
        pass


class HTPS(Search):
    def __init__(self, goal_model: GoalModel, exploration_constant=1):
        super().__init__()
        self.goal_model = goal_model
        self.exploration_constant = exploration_constant

    def reset(self, root):
        self.__init__(self.goal_model)
        self.root = root
        self.nodes[root.goal] = root

        # todo move to model
        node_goals = ['<extra_id_0>' + self.root.goal]
        # Initialise scores for root
        scores = ray.get(self.goal_model.run.remote(node_goals))

        self.root.provable_score = scores[0]
        self.root.up_score = scores[0]

    def uct(self, edge):
        return edge.logprob + self.exploration_constant * (
                self.scores[(edge.src, edge.tactic)] / edge.visit_count())  # ...

    # given a node, decide which edge to take according to search policy
    # todo update visit counts
    def expand_node(self, node):
        # if leaf, return node
        if not node.is_explored:
            return [node]

        best_score = -math.inf
        best_edge = None
        for edge in node.out_edges:
            edge_score = self.uct(edge)
            if edge_score > best_score:
                best_score = edge_score
                best_edge = edge

        ret = []
        for d in best_edge.dst:
            ret.extend(self.expand_node(d))

        return ret

    # construct a hypertree from root, until we find leaves not explored
    def get_goals(self):
        return self.expand_node(self.root)

    # compute scores for new goals, and update value estimates for tree
    def process_response(self, response):
        search_node = response.src
        result = response.dst

        # find new nodes from response, and compute their provable score
        new_nodes = []
        for result_node in result:
            if isinstance(result_node, InternalNode):
                if result_node.goal not in self.nodes:
                    new_nodes.append(result_node)
                    self.nodes[result_node.goal] = result_node

        if new_nodes:
            # todo move to model
            node_goals = ['<extra_id_0>' + node_.goal for node_ in new_nodes]

            scores = ray.get(self.goal_model.run.remote(node_goals))

            # Initialise provable_score/up_score for new internal nodes
            for i, node_ in enumerate(new_nodes):
                node_.provable_score = scores[i]
                node_.up_score = node_.provable_score
                assert self.nodes[node_.goal] is node_

        self.propagate(search_node, response)

    def propagate(self, node, response):
        result = response.dst
        score = 0
        for result_node in result:
            score += result_node.up_score
            node.visit_count += 1

        node.up_score += score

        for edge in node.in_edges:
            self.propagate(edge.src, edge)
