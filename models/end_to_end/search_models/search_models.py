from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from experiments.end_to_end.proof_node import *

"""

Abstract search model for the end-to-end proof system. All search models should inherit from this class.

"""


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


class GoalModel:
    def __init__(self, model):
        self.model = model

    def run(self, goals):
        scores = self.model.batch_generate(goals)
        return scores
