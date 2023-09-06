from data.data_modules import PremiseDataModule
from data.holist.holist_datamodule import HOListDataModule

def get_data(data_config, experiment='premise_selection'):
    if experiment == 'premise_selection':
        return PremiseDataModule(config=data_config)
    elif experiment == 'holist_pretrain':
        return HOListDataModule(config=data_config)
