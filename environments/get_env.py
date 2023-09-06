from environments.hol4.new_env import *

def get_env(config):
    if config == 'HOL4':
        return HolEnv("T")