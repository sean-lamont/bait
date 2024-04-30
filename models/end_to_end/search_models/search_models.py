from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from collections import deque

import ray
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from experiments.end_to_end.proof_node import *
from models.end_to_end.search_models.goal_model.model import SimpleGoalModel
import random
from loguru import logger


class GoalModel:
    def __init__(self, model):
        self.model = model

    def run(self, goals):
        scores = self.model.batch_generate(goals)
        return scores


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
    def process_responses(self, response: List[Edge]):
        return


# todo normalise with prior, using e.g. BestFS cumulative logprob
# todo determine how to split prior scores for siblings, e.g. divide by # siblings
# todo exploration epsilon, random selection over valid fringe scores
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


# based on cumulative logprob, maintain priority queue, pop for get_goals, populate in process_response
# currently only permits goals to be expanded once
class BestFS(Search):
    def __init__(self):
        super().__init__()
        self.priority_queue = []

        # record the chosen nodes for further analysis
        self.search_trace = []

    def reset(self, root):
        self.__init__()
        self.root = root
        if isinstance(root, InternalNode):
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

    def process_responses(self, responses: List[Edge]):
        for response in responses:
            result = response.dst

            for result_node in result:
                # Don't search proved/explored/queued nodes
                if isinstance(result_node,
                              InternalNode) and result_node not in self.priority_queue and not result_node.is_explored:
                    self.nodes[result_node.goal] = result_node
                    self.priority_queue.append(result_node)

        self.search_trace.append(responses)

        return


# Breadth First Search
class BFS(Search):
    def __init__(self):
        super().__init__()
        self.queue = deque([])

    def reset(self, root):
        self.__init__()
        self.root = root
        if isinstance(root, InternalNode):
            self.queue = deque([root])
            self.nodes[root.goal] = root

    def get_goals(self):
        if len(self.queue) > 0:
            search_node = self.queue.popleft()
            # if node was set to explored since being added (e.g. if ancestor was proven)
            if search_node.is_explored:
                return self.get_goals()

            # no score for BFS
            return [(search_node, 0.0)]
        else:
            return None

    # append new nodes to right of queue as they appear
    def process_responses(self, responses: List[Edge]):
        for response in responses:
            result = response.dst

            for result_node in result:
                # Don't search proved/explored/queued nodes
                if isinstance(result_node,
                              InternalNode) and result_node not in self.queue and not result_node.is_explored:
                    self.nodes[result_node.goal] = result_node
                    self.queue.append(result_node)

        return


# Fringe approach from TacticZero.
# Score each goal in a fringe and take the product, then take the first goal.

# Similar to BestFS without subgoal separation, except scored by goal model rather than tactic logprob
# At most one new fringe generated per tactic application. Hence UpDown is superior in that it finds all possible fringes

class FringeSearch(Search):
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


# implement as in paper, assuming n sequential expansions per goal
class HTPS(Search):
    def __init__(self, goal_model: GoalModel, exploration_constant=1):
        super().__init__()
        self.goal_model = goal_model
        self.exploration_constant = exploration_constant

        # map edges to visit counts, virtual counts, current Q estimates
        self.edge_data = {}

        # keep track of current HyperTree, refreshed after every expansion
        self.T = {}

        # record the state of the hypergraph every step for further analysis
        self.search_trace = []

        # track the leaves for score backpropagation
        self.leaves = []

    def reset(self, root):
        self.__init__(self.goal_model, self.exploration_constant)
        self.root = root
        if isinstance(root, InternalNode):
            self.nodes[root.goal] = root

        # Initialise edge_data for root
        self.edge_data = {}
        self.T = {}
        self.leaves = []

    # node score can be found by taking maximum over all edges for a given goal
    def p_uct(self, edge_dict):
        w_score = edge_dict['w_score']
        visit_count = edge_dict['visit_count']
        virtual_count = edge_dict['virtual_count']
        edge = edge_dict['edge']

        total_visits = visit_count + virtual_count
        policy_score = math.exp(edge.tac_logprob)
        node_visits = sum([self.edge_data[e]['visit_count'] for e in self.edge_data.keys() if e[0] == edge.src.goal])

        # define value estimate
        if visit_count == 0:
            q_score = 0.5 / max(1, total_visits)
        elif edge.distance_to_proof() < math.inf:
            q_score = max(1, visit_count) / max(1, total_visits)
        else:
            q_score = w_score / total_visits

        return q_score + self.exploration_constant * policy_score * (math.sqrt(node_visits) / (1 + total_visits))

    # construct a hypertree from root, until we find leaves not explored
    def get_goals(self):
        to_explore = [(self.root, None)]
        self.T = {}
        ret = []
        self.leaves = []

        # (note: cycles are automatically ignored in our tree construction)
        while to_explore:
            g, parent = to_explore.pop()

            if isinstance(g, InternalNode):
                # leaf node
                if not g.out_edges and g not in self.leaves:
                    self.leaves.append((g, parent))
                    if not g.is_explored:
                        # only return leaf nodes to expand which haven't been explored
                        ret.append((g, 0.))
                    continue

                # Expand open nodes
                if g.status != Status.FAILED:
                    # goals may appear only once in the tree todo multiple parents?
                    if g.goal not in self.T:
                        best_score = -math.inf
                        best_edge = None

                        # get the valid edges from this node, which will be edges with expandable (open/proven) children
                        # note that there must be at least valid edge, otherwise g.status == FAILED
                        goal_edges = [self.edge_data[e] for e in self.edge_data.keys() if e[0] == g.goal
                                      and any([d.status != Status.FAILED for d in self.edge_data[e]['edge'].dst])]

                        # todo check when this errors out
                        assert goal_edges, (g, self.leaves, self.T, self.edge_data)

                        for edge in goal_edges:
                            edge_score = self.p_uct(edge)
                            if edge_score > best_score:
                                best_score = edge_score
                                best_edge = edge['edge']

                        self.edge_data[(g.goal, best_edge.tactic)]['virtual_count'] += 1

                        self.T[g.goal] = {'edge': best_edge, 'parent': parent,
                                          'is_prop': False, 'uct_score': best_score}

                        # If we lead to a direct proof, then this is included as a leaf node
                        if len(best_edge.dst) == 1 and isinstance(best_edge.dst[0], ProofFinishedNode):
                            self.leaves.append((g, parent))
                        else:
                            to_explore.extend([(d, g.goal) for d in best_edge.dst])

                # if we have a Failed node
                else:
                    self.leaves.append((g, parent))

        if not ret:
            self.search_trace.append(copy.deepcopy((self.edge_data, self.T, self.leaves)))

        return ret

    def process_responses(self, responses: List):
        for response in responses:
            result = response.dst
            # find new nodes from response
            new_nodes = []
            for result_node in result:
                if isinstance(result_node, InternalNode):
                    if result_node.goal not in self.nodes:
                        new_nodes.append(result_node)
                        self.nodes[result_node.goal] = result_node

        # filter responses, taking the fastest tactic per outcome
        filtered_responses = []
        for leaf, parent in self.leaves:
            unique_dst = []
            src_filtered = []
            valid_children = [r for r in responses if r.src == leaf and all([d.status != Status.FAILED for d in r.dst])]
            for response in valid_children:
                if isinstance(response.dst[0], ProofFinishedNode):
                    response_children = 'proven'
                else:
                    response_children = set([r.goal for r in response.dst])
                if response_children not in unique_dst:
                    unique_dst.append(response_children)
                    src_filtered.append(response)
                else:
                    prev_edge = unique_dst.index(response_children)
                    if response.time < src_filtered[prev_edge].time:
                        src_filtered[prev_edge] = response

            filtered_responses.extend(src_filtered)

        # initialise scores and counts
        for edge in filtered_responses:
            self.edge_data[(edge.src.goal, edge.tactic)] = {'w_score': 0, 'visit_count': 0, 'virtual_count': 0,
                                                            'edge': edge}

        self.propagate_values()

        self.search_trace.append(copy.deepcopy((self.edge_data, self.T, self.leaves)))

    def propagate_values(self):
        if len(self.leaves) == 1 and self.leaves[0][0] == self.root:
            return

        to_backup = []

        for g, parent in self.leaves:
            if g.status == Status.PROVED:
                self.T[g.goal] = {'v_score': 1, 'parent': parent, 'is_prop': True, 'edge': None}
            elif g.status == Status.FAILED:
                self.T[g.goal] = {'v_score': 0, 'parent': parent, 'is_prop': True, 'edge': None}
            else:
                score = ray.get(self.goal_model.run.remote([g.goal]))
                self.T[g.goal] = {'v_score': math.exp(score.item()), 'parent': parent, 'is_prop': True, 'edge': None}

                # for no critic model:
                # self.T[g.goal] = {'v_score': 0.5, 'parent': parent, 'is_prop': True, 'edge': None}

            to_backup.append(parent)

        to_backup = list(set(to_backup))

        while to_backup:
            g = to_backup.pop()

            if self.T[g]['edge'] is None or self.T[g]['is_prop']:
                continue

            edge = self.T[g]['edge']
            parent = self.T[g]['parent']

            # if all children aren't propagated, continue, as they will call parent later
            if not all([self.T[child.goal]['is_prop'] for child in edge.dst]):
                continue

            to_update = 1
            for child in edge.dst:
                to_update *= self.T[child.goal]['v_score']

            self.edge_data[(g, edge.tactic)]['w_score'] += to_update
            self.edge_data[(g, edge.tactic)]['visit_count'] += 1
            self.edge_data[(g, edge.tactic)]['virtual_count'] -= 1

            self.T[g]['v_score'] = to_update
            self.T[g]['is_prop'] = True

            if parent and all([self.T[child.goal]['is_prop'] for child in self.T[parent]['edge'].dst]):
                to_backup.append(parent)


# implement as in paper, assuming n sequential expansions per goal
class TS_HTPS(Search):
    def __init__(self, goal_model: GoalModel):
        super().__init__()
        self.goal_model = goal_model

        # map edges to visit counts, virtual counts, current Q estimates
        self.edge_data = {}

        # keep track of current HyperTree, refreshed after every expansion
        self.T = {}

        # record the state of the hypergraph every step for further analysis
        self.search_trace = []

        # track the leaves for score backpropagation
        self.leaves = []

        # keep track of parameters for TS for each node
        self.ts_params = {}

    def reset(self, root):
        self.__init__(self.goal_model)
        self.root = root

        if isinstance(root, InternalNode):
            self.nodes[root.goal] = root

        # Initialise
        self.edge_data = {}
        self.T = {}
        self.leaves = []
        self.ts_params = {}

    def ts(self, edge_dict):
        edge = edge_dict['edge']

        if len(edge.dst) == 1 and isinstance(edge.dst[0], ErrorNode):
            return 0

        ts_params = [self.ts_params[d.goal] for d in edge.dst]

        # sample from beta distribution with ts_params for each child
        ts_scores = [random.betavariate(a, b) for a, b in ts_params]

        # take product of scores
        ts_score = 1
        for score in ts_scores:
            ts_score *= score

        return ts_score

    # construct a hypertree from root, until we find leaves not explored
    def get_goals(self):
        to_explore = [(self.root, None)]
        self.T = {}
        ret = []
        self.leaves = []

        # (note: cycles are automatically ignored in our tree construction)
        while to_explore:
            g, parent = to_explore.pop()

            if isinstance(g, InternalNode):
                # leaf node
                if not g.out_edges and g not in self.leaves:
                    self.leaves.append((g, parent))
                    if not g.is_explored:
                        # only return leaf nodes to expand which haven't been explored
                        ret.append((g, 0.))
                    continue

                # Expand open nodes
                if g.status != Status.FAILED:
                    # goals may appear only once in the tree todo multiple parents?
                    if g.goal not in self.T:
                        best_score = -math.inf
                        best_edge = None

                        # get the valid edges from this node, which will be edges with expandable (open/proven) children
                        # note that there must be at least valid edge, otherwise g.status == FAILED
                        goal_edges = [self.edge_data[e] for e in self.edge_data.keys() if e[0] == g.goal
                                      and any([d.status != Status.FAILED for d in self.edge_data[e]['edge'].dst])]

                        # todo check when this errors out
                        assert goal_edges, (g, g in self.leaves, g.status, g.goal, g.out_edges)

                        for edge in goal_edges:
                            edge_score = self.ts(edge)
                            if edge_score > best_score:
                                best_score = edge_score
                                best_edge = edge['edge']

                        self.T[g.goal] = {'edge': best_edge, 'parent': parent,
                                          'is_prop': False, 'ts_score': best_score}

                        # If we lead to a direct proof, then this is included as a leaf node
                        if len(best_edge.dst) == 1 and isinstance(best_edge.dst[0], ProofFinishedNode):
                            self.leaves.append((g, parent))
                        else:
                            to_explore.extend([(d, g.goal) for d in best_edge.dst])

                # if we have a Failed node
                else:
                    self.leaves.append((g, parent))

        if not ret:
            self.search_trace.append(copy.deepcopy((self.edge_data, self.T, self.leaves)))

        return ret

    def process_responses(self, responses: List):
        for response in responses:
            result = response.dst
            # find new nodes from response
            new_nodes = []
            for result_node in result:
                if isinstance(result_node, InternalNode):
                    if result_node.goal not in self.nodes:
                        new_nodes.append(result_node)
                        self.nodes[result_node.goal] = result_node

        # filter responses, taking the fastest tactic per outcome
        filtered_responses = []

        for leaf, parent in self.leaves:
            unique_dst = []
            src_filtered = []
            valid_children = [r for r in responses if r.src == leaf and all([d.status != Status.FAILED for d in r.dst])]
            for response in valid_children:
                if isinstance(response.dst[0], ProofFinishedNode):
                    response_children = 'proven'
                else:
                    response_children = set([r.goal for r in response.dst])
                if response_children not in unique_dst:
                    unique_dst.append(response_children)
                    src_filtered.append(response)
                else:
                    prev_edge = unique_dst.index(response_children)
                    if response.time < src_filtered[prev_edge].time:
                        src_filtered[prev_edge] = response

            filtered_responses.extend(src_filtered)

        # initialise scores and counts
        for edge in filtered_responses:
            self.edge_data[(edge.src.goal, edge.tactic)] = {'edge': edge}

        self.propagate_values()

        self.search_trace.append(copy.deepcopy((self.edge_data, self.T, self.leaves)))

    def propagate_values(self):
        if len(self.leaves) == 1 and self.leaves[0][0] == self.root:
            return

        to_backup = []

        for g, parent in self.leaves:
            if g.status == Status.PROVED:
                self.T[g.goal] = {'v_score': 1, 'parent': parent, 'is_prop': True, 'edge': None}
            elif g.status == Status.FAILED:
                self.T[g.goal] = {'v_score': 0, 'parent': parent, 'is_prop': True, 'edge': None}
            else:
                score = ray.get(self.goal_model.run.remote([g.goal])).item()
                alpha = score
                beta = 1 - score
                self.ts_params[g.goal] = (alpha, beta)
                # sample bernoulli prob from alpha, beta, then sample using this

                prob = random.betavariate(alpha, beta)
                value = int(random.random() < prob)

                self.T[g.goal] = {'v_score': value, 'parent': parent, 'is_prop': True, 'edge': None}

            to_backup.append(parent)

        to_backup = list(set(to_backup))

        while to_backup:
            g = to_backup.pop()

            if self.T[g]['edge'] is None or self.T[g]['is_prop']:
                continue

            edge = self.T[g]['edge']
            parent = self.T[g]['parent']

            # if all children aren't propagated, continue, as they will call parent later
            if not all([self.T[child.goal]['is_prop'] for child in edge.dst]):
                continue

            to_update = 1
            for child in edge.dst:
                to_update *= self.T[child.goal]['v_score']

            self.T[g]['v_score'] = to_update
            self.T[g]['is_prop'] = True

            if parent and all([self.T[child.goal]['is_prop'] for child in self.T[parent]['edge'].dst]):
                to_backup.append(parent)


# Thompson sampling without score model.
# Prior for each edge given by normalised (wrt out edges of source node) tactic logprob (a = lobprob, b = 1 - logprob)
# Scale factor determines how much prior changes with logprob
# Temperature is for edge normalisation (higher temperature -> more exploration)
# Depth penalty encourages exploration for deeper nodes, reducing the value of nodes further from the root
# Intuition is that if you are deeper, you are more likely to be wrong, so you should explore more

class SimpleTS(Search):
    def __init__(self, depth_penalty=0.9, scale_factor=3.0, temperature=1.0, prior=5.0):
        super().__init__()
        # map edges to visit counts, virtual counts, current Q estimates
        self.edge_data = {}

        # keep track of current HyperTree, refreshed after every expansion
        self.T = {}

        # record the state of the hypergraph every step for further analysis
        self.search_trace = []

        # track the leaves for score backpropagation
        self.leaves = []

        self.depth_penalty = depth_penalty
        self.scale_factor = scale_factor
        self.temperature = temperature
        self.prior = prior

    def reset(self, root):
        self.__init__(self.depth_penalty, self.scale_factor, self.temperature, self.prior)
        self.root = root

        if isinstance(root, InternalNode):
            self.nodes[root.goal] = root

        # Initialise
        self.edge_data = {}
        self.T = {}
        self.leaves = []

    def ts(self, edge_dict, depth):
        edge = edge_dict['edge']
        visit_count = edge_dict['visit_count']

        if len(edge.dst) == 1 and isinstance(edge.dst[0], ErrorNode):
            return 0

        a, b = edge_dict['ts_params']

        depth_penalty = self.depth_penalty ** depth

        scale = self.scale_factor * depth_penalty

        # scale a, b, then add visit count to b
        a = scale * a
        b = scale * b + visit_count

        # sample from beta distribution with ts_params for each child
        ts_score = random.betavariate(self.prior + a, self.prior + b)

        return ts_score

    # construct a hypertree from root, until we find leaves not explored
    def get_goals(self):
        to_explore = [(self.root, None, 0)]
        self.T = {}
        ret = []
        self.leaves = []

        # (note: cycles are automatically ignored in our tree construction)
        while to_explore:
            g, parent, depth = to_explore.pop()

            if isinstance(g, InternalNode):
                # leaf node
                if not g.out_edges and g not in self.leaves:
                    self.leaves.append((g, parent))
                    if not g.is_explored:
                        # only return leaf nodes to expand which haven't been explored
                        ret.append((g, 0.))
                    continue

                # Expand open nodes
                if g.status != Status.FAILED:
                    # goals may appear only once in the tree todo multiple parents?
                    if g.goal not in self.T:
                        best_score = -math.inf
                        best_edge = None

                        # get the valid edges from this node, which will be edges with expandable (open/proven) children
                        # note that there must be at least valid edge, otherwise g.status == FAILED
                        goal_edges = [self.edge_data[e] for e in self.edge_data.keys() if e[0] == g.goal
                                      and any([d.status != Status.FAILED for d in self.edge_data[e]['edge'].dst])]

                        # todo check when this errors out
                        assert goal_edges, (g, g in self.leaves, g.status, g.goal, g.out_edges)

                        for edge in goal_edges:
                            edge_score = self.ts(edge, depth)
                            if edge_score > best_score:
                                best_score = edge_score
                                best_edge = edge['edge']

                        self.T[g.goal] = {'edge': best_edge, 'ts_score': best_score}

                        # add visit count
                        self.edge_data[(g.goal, best_edge.tactic)]['visit_count'] += 1

                        # If we lead to a direct proof, then this is included as a leaf node
                        if len(best_edge.dst) == 1 and isinstance(best_edge.dst[0], ProofFinishedNode):
                            self.leaves.append((g, parent))
                        else:
                            to_explore.extend([(d, g.goal, depth + 1) for d in best_edge.dst])

                # if we have a Failed node
                else:
                    self.leaves.append((g, parent))

        if not ret:
            self.search_trace.append(copy.deepcopy((self.edge_data, self.T, self.leaves)))

        # logger.info(f'selected edges: {[e["edge"].tactic for e in self.T.values()]}')

        return ret

    def process_responses(self, responses: List):
        for response in responses:
            result = response.dst
            # find new nodes from response
            new_nodes = []
            for result_node in result:
                if isinstance(result_node, InternalNode):
                    if result_node.goal not in self.nodes:
                        new_nodes.append(result_node)
                        self.nodes[result_node.goal] = result_node

        # filter responses, taking the fastest tactic per outcome
        filtered_responses = []

        for leaf, parent in self.leaves:
            unique_dst = []
            src_filtered = []
            valid_children = [r for r in responses if r.src == leaf and all([d.status != Status.FAILED for d in r.dst])]
            for response in valid_children:
                if isinstance(response.dst[0], ProofFinishedNode):
                    response_children = 'proven'
                else:
                    response_children = set([r.goal for r in response.dst])
                if response_children not in unique_dst:
                    unique_dst.append(response_children)
                    src_filtered.append(response)
                else:
                    prev_edge = unique_dst.index(response_children)
                    if response.time < src_filtered[prev_edge].time:
                        src_filtered[prev_edge] = response

            filtered_responses.extend(src_filtered)

        # initialise scores and counts
        for edge in filtered_responses:
            self.edge_data[(edge.src.goal, edge.tactic)] = {'edge': edge, 'visit_count': 0}

        # set the TS scores for each edge, by normalising the tactic logprobs wrt the source node
        sources = set([e.src for e in filtered_responses])

        eps = 1e-5
        for s in sources:
            valid_edges = [e for e in filtered_responses if e.src == s]
            logprobs = torch.tensor([e.tac_logprob / self.temperature for e in valid_edges])
            probs = F.softmax(logprobs, dim=0)
            for i, edge in enumerate(valid_edges):
                a = max(eps, probs[i].item())
                b = max(eps, 1 - a)
                self.edge_data[(edge.src.goal, edge.tactic)]['ts_params'] = (a, b)

        self.search_trace.append(copy.deepcopy((self.edge_data, self.T, self.leaves)))


def get_search_model(config, device):
    if config.search == 'bestfs':
        return BestFS()
    elif config.search == 'bfs':
        return BFS()
    elif config.search == 'updown':
        goal_model = SimpleGoalModel.load(config.ckpt_path, device=device, freeze=True)
        if config.distributed:
            goal_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(GoalModel).remote(
                goal_model)
        else:
            goal_model = GoalModel(goal_model)
        return UpDown(goal_model)
    elif config.search == 'htps':
        goal_model = SimpleGoalModel.load(config.ckpt_path, device=device, freeze=True)

        if config.distributed:
            goal_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(GoalModel).remote(
                goal_model)
        else:
            goal_model = GoalModel(goal_model)
        return HTPS(goal_model=goal_model, exploration_constant=config.exploration_constant)
    elif config.search == 'simple_ts':
        return SimpleTS(depth_penalty=config.depth_penalty, scale_factor=config.scale_factor,
                        temperature=config.temperature, prior=config.prior)
    elif config.search == 'fringe':
        raise NotImplementedError(f'Search approach {config.search} not implemented')
    else:
        raise NotImplementedError(f'Search approach {config.search} not implemented')
