import logging
from experiments.holist.train.experiment import ExperimentRunner

if __name__ == "__main__":
    NUM_TOKENS = 2044 + 5
    logging.basicConfig(level=logging.DEBUG)

    exp_config = {
        "data_type": "sequence",
        "project": "HOList Pretrain",
        "name": "Transformer Polished",
        "learning_rate": 1e-4,
        "epochs": 20,
        "weight_decay": 1e-6,
        "batch_size": 2,
        "val_size": 512,
        "checkpoint_dir": "/home/sean/Documents/phd/deepmath-light/deepmath/train/",
        "data_dir": '/home/sean/Documents/phd/deepmath-light/deepmath/processed_train_data/',
        "device": [0],
        "val_frequency": 2048,
        "num_tactics": 41,
        "tac_embed_dim": 128,
        "final_embed_dim": 1024
    }

    embedding_model_config = {
        "model_type": "transformer",
        "vocab_size": NUM_TOKENS,
        "embedding_dim": 256,
        "dim_feedforward": 256,
        "num_heads": 4,
        "num_layers": 4,
        "dropout": 0.5,
        'small_inner': True
    }

    experiment = ExperimentRunner(exp_config=exp_config, embedding_model_config=embedding_model_config)
    experiment.run()

    exp_config['name'] = "Transformer Full Sequence"


    experiment = ExperimentRunner(exp_config=exp_config, embedding_model_config=embedding_model_config)
    experiment.run()


