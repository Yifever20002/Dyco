import torch.optim as optim

from configs import cfg
from utils import custom_print

_optimizers = {
    'adam': optim.Adam
}

def get_customized_lr_names():
    return [k[3:] for k in cfg.train.keys() if k.startswith('lr_')]

def get_optimizer(network):
    optimizer = _optimizers[cfg.train.optimizer]

    cus_lr_names = get_customized_lr_names()
    params = []
    custom_print('\n\n********** learnable parameters **********\n')
    for key, value in network.named_parameters():
        if not value.requires_grad:
            continue

        is_assigned_lr = False
        for lr_name in cus_lr_names:
            if lr_name in key:
                params += [{"params": [value], 
                            "lr": cfg.train[f'lr_{lr_name}'],
                            "name": lr_name}]
                custom_print(f"{key}: lr = {cfg.train[f'lr_{lr_name}']} shape={tuple(value.shape)}")
                is_assigned_lr = True

        if not is_assigned_lr:
            params += [{"params": [value], 
                        "name": key}]
            custom_print(f"{key}: lr = {cfg.train.lr} shape={tuple(value.shape)}")

    custom_print('\n******************************************\n\n')

    if cfg.train.optimizer == 'adam':
        optimizer = optimizer(params, lr=cfg.train.lr, betas=(0.9, 0.999))
    else:
        assert False, "Unsupported Optimizer."
        
    return optimizer
