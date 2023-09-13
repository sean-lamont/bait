from environments.hol4.env_wrapper import HolEnv as HolEnv
from environments.hol4.graph_env import HolEnv as HOL4GraphFringe
from environments.hol4.updown_env import HolEnv as HOL4UpDown

def get_env(config):
    if config == 'HOL4':
        return HolEnv("T")
    elif config == 'HOL4GraphFringe':
        return HOL4GraphFringe("T")
    elif config == 'HOL4UpDown':
        return HOL4UpDown("T")
