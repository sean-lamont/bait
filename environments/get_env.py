from environments.hol4.env_wrapper import *

def get_env(config):
    if config == 'HOL4':
        return HolEnv("T")