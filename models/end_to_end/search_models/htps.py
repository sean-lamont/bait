from __future__ import division, absolute_import, print_function

import copy
import math
from typing import List

import ray

from experiments.end_to_end.proof_node import InternalNode, Status, ProofFinishedNode
from models.end_to_end.search_models.search_models import Search, GoalModel


class HTPS(Search):
    """

    HyperTree Proof Search (HTPS) implementation

    """
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
