from environments.HOL4.tacticzero_old.env_wrapper import HolEnv as HolEnv
from environments.HOL4.tacticzero_old.graph_env import HolEnv as HOL4GraphFringe
from environments.HOL4.tacticzero_old.updown_env import HolEnv as HOL4UpDown


def get_env(config):
    if config == 'HOL4':
        return
    elif config == 'HOL4GraphFringe':
        return HOL4GraphFringe("T")
    elif config == 'HOL4UpDown':
        return HOL4UpDown("T")
