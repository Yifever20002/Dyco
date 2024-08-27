from configs import cfg
from configs.config import save_cfg

from core.utils.log_util import Logger
from core.data import create_dataloader
from core.nets import create_network
from core.train import create_trainer, create_optimizer
import wandb, os
import torch, numpy as np

# import DDP #
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from utils import get_local_rank, init_env, is_master, get_rank, get_world_size
from utils import custom_print

def init_wandb():
    wandb.login(key='yourkey')
    wandb_run = wandb.init(project='HumanNerf', config=cfg, resume=cfg.resume, dir=cfg.logdir)
    wandb.run.name = '/'.join(cfg.logdir.split('/')[-2:])
    wandb.run.save()
    return wandb_run

def main():
    log = Logger()
    log.print_config()
    wandb_run = None #init_wandb()
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    save_cfg(cfg, file_path=os.path.join(cfg.logdir,'config.yaml'))
    # initialize DDP environment
    if cfg['ddp']:
        init_env(cfg)

    model = create_network()
    optimizer = create_optimizer(model)
    trainer = create_trainer(model, optimizer, wandb_run)
    if cfg['ddp']:
        trainer = DDP(trainer, device_ids=cfg['device_ids'], output_device=get_local_rank(), find_unused_parameters=False)

    train_loader, sampler = create_dataloader('train')
    novelview_loader, _ = create_dataloader('novelview')
    novelpose_loader, _ = create_dataloader('novelpose')

    # estimate start epoch
    if cfg['ddp']:
        epoch = trainer.module.iter // len(train_loader) + 1
    else:
        epoch = trainer.iter // len(train_loader) + 1
    while True:
        if cfg['ddp']:
            if trainer.module.iter > cfg.train.maxiter:
                break
        else:
            if trainer.iter > cfg.train.maxiter:
                break     

        if epoch == 1 or epoch%cfg['val_interval'] == 0:
            trainer.module.validate(render_folder_name='novelview', val_dataloader=novelview_loader)
            trainer.module.validate(render_folder_name='novelpose', val_dataloader=novelpose_loader)

        if cfg['ddp']:
            sampler.set_epoch(epoch)
            trainer.module.train(epoch=epoch,
                        train_dataloader=train_loader)
        else:
            trainer.train(epoch=epoch,
                        train_dataloader=train_loader)
        epoch += 1

    trainer.module.finalize()

if __name__ == '__main__':
    main()
