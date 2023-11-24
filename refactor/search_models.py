from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from loguru import logger
import pytorch_lightning as pl

import ray

from refactor.goal_model.model import SimpleGoalModel
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
    def process_responses(self, response: List[Edge]):
        return


class UpDown(Search):
    def __init__(self, goal_model: GoalModel):
        super().__init__()
        self.goal_model = goal_model

    def reset(self, root):
        self.__init__(self.goal_model)
        self.root = root
        if isinstance(root, InternalNode):
            self.nodes[root.goal] = root

            # Initialise scores for root
            scores = ray.get(self.goal_model.run.remote([self.root.goal]))

            self.root.provable_score = scores[0].item()
            self.root.up_score = self.root.provable_score

    def get_goals(self):
        best_score = -math.inf
        best_node = None

        node_scores = {}
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

            node_scores[node.goal] = score
            if score > best_score:
                best_score = score
                best_node = node

        if not best_node:
            return []

        # todo take as parameter # of fringes then sample? Would take set of all scores, then sample from the set and
        # choose fringe with the score as below

        # find fringe for selected node by choosing all goals with the same score.
        # (may include other goals with same score not in fringe)
        best_fringe = []

        for goal, score in node_scores.items():
            if score == best_score:
                best_fringe.append((self.nodes[goal], best_score))

        return best_fringe

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
    def process_responses(self, responses: List[Edge]):
        for response in responses:
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
                scores = ray.get(self.goal_model.run.remote([g.goal for g in new_nodes]))

                # Initialise provable_score/up_score for new internal nodes
                for i, node_ in enumerate(new_nodes):
                    node_.provable_score = (scores[i] + (node_.depth * math.log(0.99))).item()
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

        return


# Similar to BestFS, but score each goal in a state and take the product, then take the first goal
class FringeSearch(Search):
    def get_goals(self):
        pass

    def process_responses(self, response):
        pass

    def reset(self, root):
        pass


class BFS(Search):
    def reset(self, root):
        pass

    def get_goals(self):
        pass

    def process_responses(self, response):
        pass


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

    def reset(self, root):
        self.__init__(self.goal_model, self.exploration_constant)
        self.root = root
        if isinstance(root, InternalNode):
            self.nodes[root.goal] = root

        # Initialise edge_data for root
        self.edge_data = {}

    # node score can be found by taking maximum over all edges for a given goal
    def p_uct(self, edge_data):
        w_score = edge_data['w_score']
        visit_count = edge_data['visit_count']
        virtual_count = edge_data['virtual_count']
        edge = edge_data['edge']

        total_visits = visit_count + virtual_count
        policy_score = edge.tac_logprob
        node_visits = sum([self.edge_data[e]['visit_count'] for e in self.edge_data.keys() if e[0] == edge.src.goal])
        # node_visits = sum([self.edge_data[(edge.src.goal, edge_.tactic)]['visit_count'] for edge_ in self.edge_data])

        # define value estimate
        if visit_count == 0:
            q_score = 0.5 / (max(1, total_visits))
        elif edge.distance_to_proof() < math.inf:
            q_score = max(1, visit_count) / max(1, total_visits)
        else:
            q_score = w_score / total_visits

        return q_score + self.exploration_constant * policy_score * (math.sqrt(node_visits) / (1 + total_visits))

    # construct a hypertree from root, until we find leaves not explored
    def get_goals(self):
        to_explore = [self.root]
        self.T = {}
        leaves = []

        self.T[self.root.goal] = {'edge': None, 'parent': None,
                                  'is_prop': True, 'v_score': -math.inf}

        # (note: cycles are automatically ignored in tree construction)
        while to_explore:
            g = to_explore.pop()
            # leaf node
            if not g.out_edges:
                leaves.append((g, 0.))
                continue

            # Internal nodes are always expandable, since by definition they have an open descendant, with no cycles
            if isinstance(g, InternalNode) and g.status == Status.OPEN:
                best_score = -math.inf
                best_edge = None
                goal_edges = [self.edge_data[e] for e in self.edge_data.keys() if e[0] == g.goal]
                for edge in goal_edges:
                    edge_score = self.p_uct(edge)
                    if edge_score > best_score:
                        best_score = edge_score
                        best_edge = edge['edge']

                self.edge_data[(g.goal, best_edge.tactic)]['virtual_count'] += 1

                # goals may appear only once in the tree
                for d in best_edge.dst:
                    # todo why is error node here??? should not be added to edge data
                    if d.goal not in self.T:
                        self.T[d.goal] = {'edge': best_edge, 'parent': g.goal,
                                          'is_prop': False, 'v_score': -math.inf}
                        to_explore.append(d)

        return leaves

    # leaves are the most recently expanded nodes, only need to filter and initialise edges
    def process_responses(self, responses: List[Edge]):
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
        leaves = set([edge.src for edge in responses])

        filtered_responses = []
        for leaf in leaves:
            unique_dst = []
            src_filtered = []
            for response in [r for r in responses if r.src == leaf and r.dst[0].status == Status.OPEN]:
                if response.dst not in unique_dst:
                    unique_dst.append(response.dst)
                    src_filtered.append(response)
                else:
                    prev_edge = unique_dst.index(response.dst)
                    if response.time < src_filtered[prev_edge].time:
                        src_filtered[prev_edge] = response

            filtered_responses.extend(src_filtered)

        # initialise scores and counts
        for edge in filtered_responses:
            self.edge_data[(edge.src.goal, edge.tactic)] = {'w_score': 0, 'visit_count': 0, 'virtual_count': 0, 'edge': edge}

        self.propagate_values(leaves)

        self.search_trace.append(self.edge_data)

    def propagate_values(self, leaf_nodes):
        to_backup = []
        for g in leaf_nodes:
            if g.status == Status.PROVED:
                self.T[g.goal]['v_score'] = 1
            elif g.status == Status.FAILED:
                self.T[g.goal]['v_score'] = 0
            else:
                self.T[g.goal]['v_score'] = ray.get(self.goal_model.run.remote([g.goal]))
            to_backup.append(g.goal)

        while to_backup:
            g = to_backup.pop()
            # root
            if self.T[g]['edge'] is None:
                continue
            edge = self.T[g]['edge']
            parent = self.T[g]['parent']

            to_update = 0
            for child in edge.dst:
                to_update += self.T[child.goal]['v_score']

            # todo key error...
            self.edge_data[(g, edge.tactic)]['w_score'] += to_update
            self.edge_data[(g, edge.tactic)]['visit_count'] += 1
            self.edge_data[(g, edge.tactic)]['virtual_count'] -= 1

            self.T[g]['v_score'] = to_update
            self.T[g]['is_prop'] = True

            if parent and all([self.T[child.goal]['is_prop'] for child in self.T[parent]['edge'].dst]):
                to_backup.append(parent)


def get_search_model(config, device):
    if config.search == 'bestfs':
        return BestFS()
    elif config.search == 'bfs':
        pass
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
    else:
        raise NotImplementedError(f'Search approach {config.search} not implemented')
