from __future__ import division, absolute_import, print_function

from collections import deque
from typing import List

from experiments.end_to_end.proof_node import InternalNode, Edge
from models.end_to_end.search_models.search_models import Search


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
