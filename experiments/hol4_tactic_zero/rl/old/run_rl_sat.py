from data.hol4.ast_def import graph_to_torch_labelled
import pickle
from data.hol4 import ast_def
from environments.hol4.new_env import *
import warnings

from global_config import GLOBAL_PATH

warnings.filterwarnings('ignore')
from experiments.hol4.rl.rl_experiment import RLExperiment

# for debugging
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


####################################################################################################
# SAT
####################################################################################################

def run_rl_sat():
    premise_db = {}

    print("Generating premise graph db...")
    for i, t in enumerate(compat_db):
        premise_db[t] = graph_to_torch_labelled(ast_def.goal_to_graph_labelled(t), token_enc)

    VOCAB_SIZE = 1004
    EMBEDDING_DIM = 256

    model_config = {
    "model_type": "sat",
    # 'gnn_type': 'di_gcn',
    "num_edge_features":  200,
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 256,
    "dim_feedforward": 256,
    "num_heads": 4,
    "num_layers": 2,
    "in_embed": True,
    "se": "formula-net",
    "abs_pe": False,
    "abs_pe_dim": 256,
    "use_edge_attr": True,
    "dropout": 0.,
    "gnn_layers": 3,
    "directed_attention": False,
}

    sat_config = default_config
    sat_config['name'] = 'SAT 50 Step'
    sat_config['model_config'] = model_config
    sat_config[
        'pretrain_ckpt'] = GLOBAL_PATH + "experiments/hol4/supervised/model_checkpoints/sat_fn.ckpt"
    sat_config['exp_type'] = 'sat'
    sat_config['data_type'] = 'sat'
    sat_config['vocab_size'] = VOCAB_SIZE
    sat_config['notes'] = 'sat_50_step_new/'
    sat_config['graph_db'] = premise_db
    sat_config['embedding_dim'] = EMBEDDING_DIM

    sat_config['resume'] = True
    sat_config['pretrain'] = False
    sat_config['resume_id'] = 'azf3xowr'

    experiment = RLExperiment(sat_config)
    experiment.run()



if __name__ == '__main__':
    with open("data/hol4/data/valid_goals_shuffled.pk", "rb") as f:
        valid_goals = pickle.load(f)

    train_goals = valid_goals[:int(0.8 * len(valid_goals))]
    test_goals = valid_goals[int(0.8 * len(valid_goals)):]

    with open("data/hol4/data/graph_token_encoder.pk", "rb") as f:
        token_enc = pickle.load(f)

    encoded_graph_db = []
    with open('data/hol4/data/adjusted_db.json') as f:
        compat_db = json.load(f)

    reverse_database = {(value[0], value[1]): key for key, value in compat_db.items()}

    MORE_TACTICS = True
    if not MORE_TACTICS:
        thms_tactic = ["simp", "fs", "metis_tac"]
        thm_tactic = ["irule"]
        term_tactic = ["Induct_on"]
        no_arg_tactic = ["strip_tac"]
    else:
        thms_tactic = ["simp", "fs", "metis_tac", "rw"]
        thm_tactic = ["irule", "drule"]
        term_tactic = ["Induct_on"]
        no_arg_tactic = ["strip_tac", "EQ_TAC"]

    tactics = {
        'thms_tactic': thms_tactic,
        'thm_tactic': thm_tactic,
        'term_tactic': term_tactic,
        'no_arg_tactic': no_arg_tactic,
        'tactic_pool': thms_tactic + thm_tactic + term_tactic + no_arg_tactic
    }

    dir = GLOBAL_PATH + 'experiments/hol4/rl/lightning_rl/experiments/'

    default_config = {
        'model_config': None,
        'project': 'RL Test',
        'name': None,  # 'Relation 50 Step',
        'tactics': tactics,
        'pretrain_ckpt': None,
        'exp_type': None,
        'vocab_size': None,
        'embedding_dim': None,
        'resume': False,
        'resume_id': None,
        'pretrain': True,
        'gnn_layers': 3,
        'notes': None,  # 'x_50_step/',
        'dir': dir,
        'max_steps': 50,
        'gamma': 0.99,
        'lr': 5e-5,
        'arg_len': 5,
        'data_type': None,  # 'graph' 'relation' 'sequence'
        'train_goals': train_goals,
        'test_goals': test_goals,
        'token_enc': token_enc,
        'graph_db': None,
        'device': [0],
        'val_freq': 5,
        'reverse_database': reverse_database
    }

    torch.set_float32_matmul_precision('high')

    # import cProfile
    # cProfile.run('run_rl_gnn()', sort='cumtime')

    run_rl_sat()
    # run_rl_transformer()