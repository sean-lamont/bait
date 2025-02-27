from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import warnings

from models.end_to_end.tactic_models.tac_models import TacModel

warnings.filterwarnings('ignore')


class TopKTacGenerator(TacModel):
    def __init__(self, config):
        super().__init__()
        self.num_filtered = config.num_filtered
        self.random = config.random
        self.tac_model = config.tac_model # should be instantiated with Hydra

    def get_tactics(self, goal, premises):
        _, theorem, _ = premises
        tactics = self.tac_model.get_tactics(goal, premises)

        goal.data['original_tacs'] = tactics

        if self.random:
            return random.sample(tactics, self.num_filtered)
        else:
            # tactics are expected to be sorted here
            return tactics[:self.num_filtered]
