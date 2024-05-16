from __future__ import division, absolute_import, print_function

import copy
import math
import random
from typing import List

import torch
from torch.nn import functional as F

from experiments.end_to_end.proof_node import InternalNode, ErrorNode, Status, ProofFinishedNode
from models.end_to_end.search_models.search_models import Search


class SimpleTS(Search):
    """

    Thompson sampling without score model.
    Prior for each edge given by normalised (wrt out edges of source node) tactic logprob (a = lobprob, b = 1 - logprob)
    Scale factor determines how much prior changes with logprob
    Temperature is for edge normalisation (higher temperature -> more exploration)
    Depth penalty encourages exploration for deeper nodes, reducing the value of nodes further from the root
    Intuition is that if you are deeper, you are more likely to be wrong, so you should explore more

    """

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
