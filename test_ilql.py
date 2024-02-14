from omegaconf import OmegaConf

class MakeObj(object):
    """ dictionary to object.
    Thanks to https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object

    Args:
        object ([type]): [description]
    """
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [MakeObj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, MakeObj(b) if isinstance(b, dict) else b)


def read_yaml(path):
    x_dict =  OmegaConf.load(path)
    x_yamlstr = OmegaConf.to_yaml(x_dict)
    x_obj = MakeObj(x_dict)
    return x_yamlstr, x_dict, x_obj

x_yamlstr, x_dict, config = read_yaml('configs/end_to_end/train/ilql/run.yaml')
#%%
from hydra.utils import instantiate
model = instantiate(config.experiment)
#%%
# model.batch_generate(['x : ℝ\n⊢ log |x| = log x', 'n : ℕ∞,\nh : n ≤ 0\n⊢ ↑0 - ↑n ≤ ↑(0 - n)'], None, 2)
model.batch_generate(['abcdefg', 'abcdefghi'], None, 3)
