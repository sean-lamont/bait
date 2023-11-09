from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq

import ray

from refactor.proof_node import *


class Search:
    def __init__(self):
        self.search_trace = []
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


class GoalModel:
    def __init__(self, model):
        self.model = model

    def run(self, goals):
        scores = self.model.batch_generate(goals)
        return scores


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

        if best_node:
            self.search_trace.append((best_node.goal, best_score))
        else:
            self.search_trace.append((None, -math.inf))
            return None

        # todo pass whole fringe for running (need to get argmax context)
        # return [best_node] + [self.nodes[ctx] for ctx in best_node.context]

        return [best_node]

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

            # todo scale breadth as it's explored
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
class BestFS(Search):
    def __init__(self):
        super().__init__()
        self.priority_queue = []

    def reset(self, root):
        self.__init__()
        self.priority_queue = [root]

    def get_goals(self):
        return heapq.heappop(self.priority_queue)

    def process_response(self, response: Edge):
        result = response.dst

        for result_node in result:
            # Record the new node and add it to the search queue.
            if isinstance(result_node, InternalNode):
                # self.nodes[result_node.goal] = result_node
                pass

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
    def reset(self, root):
        pass

    def get_goals(self):
        pass

    def process_response(self, response):
        pass
