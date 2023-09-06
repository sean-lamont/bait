from experiments.hol4.rl.agent_utils import *
from experiments.hol4.rl.rl_data_module import *
import torch.optim
import pickle
from environments.hol4.new_env import *
import warnings
from global_config import GLOBAL_PATH

warnings.filterwarnings('ignore')
from experiments.hol4.rl.rl_experiment import RLExperiment

# for debugging
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

####################################################################################################
# TRANSFORMER
####################################################################################################

def run_rl_transformer():
    print("Generating Premise DB..")
    premise_db = {}

    def to_sequence(data, vocab):
        data = data.split(" ")
        data = torch.LongTensor([vocab[c] for c in data])
        return data

    vocab = {}
    i = 1
    for k in compat_db.keys():
        tmp = k.split(" ")
        for t in tmp:
            if t not in vocab:
                vocab[t] = i
                i += 1

    for i, t in enumerate(compat_db):
        premise_db[t] = to_sequence(t, vocab)

    premise_db = premise_db, vocab

    VOCAB_SIZE = 1300
    EMBEDDING_DIM = 256

    model_config = {
        "model_type": "transformer",
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "dim_feedforward": 512,
        "num_heads": 8,
        "num_layers": 4,
        "dropout": 0.0
    }

    transformer_config = default_config
    transformer_config['name'] = 'Transformer 50 Step'
    transformer_config['model_config'] = model_config
    transformer_config[
        'pretrain_ckpt'] = GLOBAL_PATH + "experiments/hol4/supervised/model_checkpoints/transformer_90_04.ckpt"
    transformer_config['exp_type'] = 'sequence'
    transformer_config['data_type'] = 'sequence'
    transformer_config['vocab_size'] = VOCAB_SIZE
    transformer_config['notes'] = 'transformer_50_step_new/'
    transformer_config['graph_db'] = premise_db
    transformer_config['embedding_dim'] = EMBEDDING_DIM

    transformer_config['resume'] = True
    transformer_config['pretrain'] = False
    transformer_config['resume_id'] = '3fy5csaj'

    experiment = RLExperiment(transformer_config)
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
        'device': [1],
        'val_freq': 5,
        'reverse_database': reverse_database
    }

    torch.set_float32_matmul_precision('high')

    run_rl_transformer()
