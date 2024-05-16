from __future__ import division, absolute_import, print_function

from math import log
from typing import List

from experiments.end_to_end.proof_node import InternalNode, Edge
from models.end_to_end.search_models.search_models import Search


# Same as BestFS, but scales cumulative logprob by the depth of the node
class LevinSearch(Search):
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
        self.priority_queue = sorted(self.priority_queue, key=lambda x: x.cumulative_logprob - log(max(x.depth, 1)))
        if len(self.priority_queue) > 0:
            search_node = self.priority_queue.pop()
            # if node was set to explored since being added (e.g. if ancestor was proven)
            if search_node.is_explored:
                return self.get_goals()

            return [(search_node, search_node.cumulative_logprob - log(max(search_node.depth, 1)))]
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
