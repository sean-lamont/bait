from data.data_modules import PremiseDataModule
from data.holist.graph_datamodule import HOListDataModule
from experiments.pyrallis_configs_old import DataConfig

def get_data(data_config: DataConfig, experiment='premise_selection'):
    if experiment == 'premise_selection':
        return PremiseDataModule(config=data_config)
    elif experiment == 'holist_pretrain':
        return HOListDataModule(config=data_config)
