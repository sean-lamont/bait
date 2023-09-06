from global_config import GLOBAL_PATH
from experiments.pretrain import SeparateEncoderPremiseSelection


# todo more accurate edge labelling based on order (e.g. =, and, or iff) given same label, otherwise in order of arguments
# todo also in daglstm paper, concatenate this with parent type?


# MIZAR vocab
VOCAB_SIZE = 13420

# HOL4 vocab
# VOCAB_SIZE = 1004

# HOLStep vocab
# VOCAB_SIZE = 1909 + 4

# HOL4 transformer
# VOCAB_SIZE = 1300

gcn_config = {
    'model_type': 'di_gcn',
    'vocab_size': VOCAB_SIZE,
    'embedding_dim': 256,
    'gnn_layers': 3
}

sat_config = {
    "model_type": "sat",
    "batch_norm" : True,
    "global_pool": "mean",
    # 'gnn_type': 'di_gcn',
    "num_edge_features": 20,
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 128,
    "dim_feedforward": 256,
    "num_heads": 2,
    "num_layers": 2,
    "in_embed": True,
    "se": "formula-net",
    # "se": "pna",
    "abs_pe": False,
    "abs_pe_dim": 256,
    "use_edge_attr": True,
    "dropout": 0.,
    "gnn_layers": 2,
    'small_inner': False,
}

transformer_config = {
    "model_type": "transformer",
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 256,
    "dim_feedforward": 512,
    "num_heads": 8,
    "num_layers": 4,
    "dropout": 0.0,
    "small_inner": True
}

relation_config = {
    # "model_type": "transformer_relation",
    "model_type": "transformer_relation_small",
    "vocab_size": VOCAB_SIZE,
    # "vocab_size": VOCAB_SIZE + 1,
    "embedding_dim": 256,
    "dim_feedforward": 512,
    "num_heads": 8,
    "num_layers": 4,
    "dropout": 0.0
}

amr_config = {
    "model_type": "amr",
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 128,
    "dim_feedforward": 512,
    "num_heads": 4,
    "num_layers": 4,
    "in_embed": True,
    "abs_pe": False,
    "abs_pe_dim": 2,
    "use_edge_attr": True,
    "device": "cuda:0",
    "dropout": 0.,
}

exp_config = {
    "experiment": "premise_selection",
    "learning_rate": 1e-4,
    "epochs": 30,
    "weight_decay": 1e-6,
    "batch_size": 32,
    "model_save": False,
    "val_size": 4096,
    "logging": False,
    "checkpoint_dir": GLOBAL_PATH + "experiments/hol4/supervised/model_checkpoints",
    # "checkpoint_dir": "/home/sean/Documents/phd/aitp/experiments/hol4/supervised/model_checkpoints",
    "device": [0],
    "max_errors": 1000,
    "val_frequency": 2048,
}

formula_net_config = {
    "model_type": "formula-net-edges",
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 128,
    "gnn_layers": 2,
    "batch_norm": True
}

digae_config = {
    "model_type": "digae",
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 256
}

gnn_transformer_config = {
    "batch_norm": True,
    "model_type": "gnn_transformer",
    "global_pool": "cls",
    "num_edges": 20,
    "edge_dim": 32,
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 128,
    "dim_feedforward": 256,
    "num_heads": 2,
    "num_layers": 2,
    "abs_pe": True,
    "dropout": 0.,
    "gnn_layers": 2,
}

h5_data_config = {"source": "h5", "data_dir": GLOBAL_PATH + "data/utils/holstep_full"}
# h5_data_config = {"source": "h5", "data_dir": "/home/sean/Documents/phd/aitp/data/utils/processed_data"}

# hol4 for relations
hol4_data_config = {"source": "hol4", "data_dir": GLOBAL_PATH + "data/hol4/torch_data"}
# hol4_graph_data_config = {"source": "hol4_graph", "data_dir": "/home/sean/Documents/phd/aitp/data/hol4/graph_torch_data"}
hol4_graph_data_config = {"source": "hol4_graph", "data_dir": GLOBAL_PATH + "data/hol4/graph_attention_data_new"}
hol4_sequence_data_config = {"source": "hol4_sequence", "data_dir": GLOBAL_PATH + "data/hol4/sequence_torch_data"}

mizar_data_config = {"source": "mizar", "data_dir": GLOBAL_PATH + 'data/mizar'}

relation_att_exp = SeparateEncoderPremiseSelection(config={"model_config": relation_config,
                                                           "exp_config": exp_config,
                                                           "data_config": h5_data_config,
                                                           "project": "test_project",
                                                           "notes": "",
                                                           "name": "relation inner small"})

# todo with original sequence for positional encoding
transformer_experiment = SeparateEncoderPremiseSelection(config={"model_config": transformer_config,
                                                                 "exp_config": exp_config,
                                                                 "data_config": hol4_sequence_data_config,
                                                                 "project": "hol4_premise_selection",
                                                                 "notes": "",
                                                                 "name": "Transformer Small Inner + Max Pool"})

sat_exp = SeparateEncoderPremiseSelection(config={"model_config": sat_config,
                                                  "exp_config": exp_config,
                                                  "data_config": mizar_data_config,
                                                  "project": "mizar_40_premise_selection",
                                                  "notes": "",
                                                  "name": "SAT"})

gcn_exp = SeparateEncoderPremiseSelection(config={"model_config": gcn_config,
                                                  "exp_config": exp_config,
                                                  "data_config": hol4_graph_data_config,
                                                  "project": "hol4_premise_selection",
                                                  "notes": "",
                                                  "name": "DiGCN"})

formula_net_exp = SeparateEncoderPremiseSelection(config={"model_config": formula_net_config,
                                                          "exp_config": exp_config,
                                                          "data_config": mizar_data_config,
                                                          "project": "mizar_40_premise_selection",
                                                          "notes": "",
                                                          "name": "FormulaNet Default + LR 5,0.5"})

digae_exp = SeparateEncoderPremiseSelection(config={"model_config": digae_config,
                                                    "exp_config": exp_config,
                                                    "data_config": h5_data_config,
                                                    "project": "test_project",
                                                    "notes": "",
                                                    "name": "digae_large"})

amr_exp = SeparateEncoderPremiseSelection(config={"model_config": amr_config,
                                                  "exp_config": exp_config,
                                                  "data_config": h5_data_config,
                                                  "project": "test_project",
                                                  "name": "amr"})

gnn_transformer_exp = SeparateEncoderPremiseSelection(config={"model_config": gnn_transformer_config,
                                                              "exp_config": exp_config,
                                                              "data_config": mizar_data_config,
                                                              "notes": "",
                                                              "project": "mizar_40_premise_selection",
                                                              "name": "GNN + Transformer Readout"})

# import cProfile

# cProfile.run('sat_exp.run_lightning()', sort = 'cumtime')


# gnn_transformer_exp.run_lightning()

# gcn_exp.run_lightning()
# amr_exp.run_lightning()
formula_net_exp.run_lightning()
# sat_exp.run_lightning()
# relation_att_exp.run_lightning()
# transformer_experiment.run_lightning()
# digae_exp.run_lightning()

