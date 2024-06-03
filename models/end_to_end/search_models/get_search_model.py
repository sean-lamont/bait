from __future__ import division, absolute_import, print_function

import ray

from models.end_to_end.search_models.bestfs import BestFS
from models.end_to_end.search_models.bfs import BFS
from models.end_to_end.search_models.goal_models.pair_model.model import PairGoalModel
from models.end_to_end.search_models.htps import HTPS
from models.end_to_end.search_models.levin_search import LevinSearch
from models.end_to_end.search_models.search_models import GoalModel
from models.end_to_end.search_models.simple_ts import SimpleTS
from models.end_to_end.search_models.updown import UpDown

from models.end_to_end.search_models.goal_models.hard_goal_model.model import HardGoalModel


def get_search_model(config, device):
    """

    Initialise search model based on configuration. Models which require a goal model should
    inherit from the GoalModel class, and set up distributed inference with ray.

    """

    if config.search == 'bestfs':
        return BestFS()
    elif config.search == 'bfs':
        return BFS()
    elif config.search == 'updown':
        goal_model = HardGoalModel.load(config.ckpt_path, device=device, freeze=True)
        if config.distributed:
            goal_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(GoalModel).remote(
                goal_model)
        else:
            goal_model = GoalModel(goal_model)
        return UpDown(goal_model)
    elif config.search == 'htps':
        goal_model = PairGoalModel.load(config.ckpt_path, device=device, freeze=True)

        if config.distributed:
            goal_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(GoalModel).remote(
                goal_model)
        else:
            goal_model = GoalModel(goal_model)
        return HTPS(goal_model=goal_model, exploration_constant=config.exploration_constant)
    elif config.search == 'simple_ts':
        return SimpleTS(depth_penalty=config.depth_penalty, scale_factor=config.scale_factor,
                        temperature=config.temperature, prior=config.prior)
    elif config.search == 'levin':
        return LevinSearch()
    elif config.search == 'fringe':
        raise NotImplementedError(f'Search approach {config.search} not implemented')
    else:
        raise NotImplementedError(f'Search approach {config.search} not implemented')
