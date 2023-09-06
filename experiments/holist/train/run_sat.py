import logging
from experiments.holist.train.experiment import ExperimentRunner

if __name__ == "__main__":
    NUM_TOKENS = 2044 + 5
    logging.basicConfig(level=logging.DEBUG)

    exp_config = {
        'data_type': 'graph',
        "project": "HOList Pretrain",
        "name": "SAT",
        "learning_rate": 1e-4,
        "epochs": 10,
        "weight_decay": 1e-6,
        "batch_size": 16,
        "val_size": 512,
        "checkpoint_dir": "/home/sean/Documents/phd/deepmath-light/deepmath/train/",
        "data_dir": '/home/sean/Documents/phd/deepmath-light/deepmath/processed_train_data/',
        "device": [1],
        "val_frequency": 2048,
        "num_tactics": 41,
        "tac_embed_dim": 128,
        "final_embed_dim": 1024
    }

    embedding_model_config = {
        "model_type": "sat",
        # 'gnn_type': 'di_gcn',
        "num_edge_features": 3,
        "vocab_size": NUM_TOKENS,
        "embedding_dim": 128,
        "dim_feedforward": 256,
        "num_heads": 4,
        "num_layers": 4,
        "in_embed": True,
        "se": "gnn-encoder",
        "abs_pe": False,
        "abs_pe_dim": 256,
        "use_edge_attr": True,
        "dropout": 0.2,
        "gnn_layers": 1,
        'small_inner': False
    }

    experiment = ExperimentRunner(exp_config=exp_config, embedding_model_config=embedding_model_config)
    experiment.run()
    #
    #
    # embedding_model_config['num_heads'] = 2
    # embedding_model_config['num_layers'] = 4
    #
    # experiment = ExperimentRunner(exp_config=exp_config, embedding_model_config=embedding_model_config)
    # experiment.run()
    #
    #
    # embedding_model_config['num_heads'] = 2
    # embedding_model_config['num_layers'] = 8
    #
    # experiment = ExperimentRunner(exp_config=exp_config, embedding_model_config=embedding_model_config)
    # experiment.run()
    #
    #
    # embedding_model_config['num_heads'] = 2
    # embedding_model_config['num_layers'] = 4
    # embedding_model_config['gnn_layers'] = 0
    #
    #
    # experiment = ExperimentRunner(exp_config=exp_config, embedding_model_config=embedding_model_config)
    # experiment.run()
    #
    # embedding_model_config['num_heads'] = 1
    # embedding_model_config['num_layers'] = 1
    # embedding_model_config['gnn_layers'] = 0
    #
    # experiment = ExperimentRunner(exp_config=exp_config, embedding_model_config=embedding_model_config)
    # experiment.run()
    #
    #
    # embedding_model_config['num_heads'] = 1
    # embedding_model_config['num_layers'] = 1
    # embedding_model_config['gnn_layers'] = 12
    #
    # experiment = ExperimentRunner(exp_config=exp_config, embedding_model_config=embedding_model_config)
    # experiment.run()
    #
    # embedding_model_config['num_heads'] = 1
    # embedding_model_config['num_layers'] = 1
    # embedding_model_config['gnn_layers'] = 4
    # embedding_model_config['dropout'] = 0
    #
    # experiment = ExperimentRunner(exp_config=exp_config, embedding_model_config=embedding_model_config)
    # experiment.run()
    #
    # embedding_model_config['num_heads'] = 1
    # embedding_model_config['num_layers'] = 1
    # embedding_model_config['gnn_layers'] = 4
    # embedding_model_config['dropout'] = 0.2
    #
    # experiment = ExperimentRunner(exp_config=exp_config, embedding_model_config=embedding_model_config)
    # experiment.run()
    #
    # embedding_model_config['num_heads'] = 2
    # embedding_model_config['num_layers'] = 2
    # embedding_model_config['gnn_layers'] = 4
    # embedding_model_config['dropout'] = 0.2
    #
    # experiment = ExperimentRunner(exp_config=exp_config, embedding_model_config=embedding_model_config)
    # experiment.run()
    #
    #
    #
