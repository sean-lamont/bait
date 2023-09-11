# from environments.hol4.env_wrapper import *
from environments.hol4.graph_env import *

def get_env(config):
    if config == 'HOL4':
        return HolEnv("T")