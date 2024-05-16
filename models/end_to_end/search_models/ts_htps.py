from __future__ import division, absolute_import, print_function

import copy
import math
import random
from typing import List

import ray

from experiments.end_to_end.proof_node import InternalNode, ErrorNode, Status, ProofFinishedNode
from models.end_to_end.search_models.search_models import Search, GoalModel


class TS_HTPS(Search):
    """

    Thompson Sampling HTPS

    """
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
