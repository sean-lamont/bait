import torch

def get_optim(config):
    optim = getattr(torch.optim, config.optimiser)

    if config.schedule:
        schedule = getattr(torch.optim.lr_scheduler, config.schedule)
        return optim,  schedule

    else:
        return optim
