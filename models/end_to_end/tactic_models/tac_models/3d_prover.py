from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import ray

from models.end_to_end.tactic_models.tac_models_ import TacModel

warnings.filterwarnings('ignore')


class DiversityTacGenerator(TacModel):
    def __init__(self, config):
        super().__init__()
        self.tac_model = config.tac_model
        self.filter_model = config.filter_model
        self.num_filtered = config.num_filtered
        self.temperature = config.temperature
        self.scale = config.scale
        self.p = config.p


def get_tactics(self, goal, premises):
    _, theorem, _ = premises

    # t0 = time.monotonic()
    tactics = self.tac_model.get_tactics(goal, premises)
    # logger.warning(f"Time to get tactics: {time.monotonic() - t0}")

    goal.data['original_tacs'] = tactics

    state = goal.data['augmented_state'] if hasattr(goal, 'data') and 'augmented_state' in goal.data else goal.goal
    # filter with filter_model

    # t0 = time.monotonic()
    inds, sim_matrix = ray.get(self.filter_model.filter_tacs.remote(tactics, self.num_filtered,
                                                                    state=state, theorem=theorem.full_name,
                                                                    temperature=self.temperature, scale=self.scale,
                                                                    p=self.p))
    # logger.warning(f"Time to filter tactics: {time.monotonic() - t0}")

    goal.data['similarity_scores'] = sim_matrix

    return [tactics[i] for i in sorted(inds[0])]
