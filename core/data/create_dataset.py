import os
import imp
import time

import numpy as np
import torch

from core.utils.file_util import list_files
from configs import cfg
from .dataset_args import DatasetArgs

# import DDP #
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


def _query_dataset(data_type):
    module = cfg[data_type].dataset_module
    module_path = module.replace(".", "/") + ".py"
    dataset = imp.load_source(module, module_path).Dataset
    return dataset


def _get_total_train_imgs(dataset_path):
    train_img_paths = \
        list_files(os.path.join(dataset_path, 'images'),
                                exts=['.png'])
    if len(train_img_paths)>0:
        return len(train_img_paths)
    else: #v2
        import pickle
        with open(os.path.join(dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return len(cameras)



def create_dataset(data_type='train',**kwargs):
    dataset_name = cfg[data_type].dataset
    args = DatasetArgs.get(dataset_name) # e.g. zju_387_testset
    # customize dataset arguments according to dataset type
    args['bgcolor'] = None if data_type == 'train' else cfg.bgcolor
    if data_type in ['progress','movement', 'novelview', 'novelpose']:
        total_train_imgs = _get_total_train_imgs(args['dataset_path'])
        if data_type == 'progress':
            if not 'skip' in args:
                args['skip'] = total_train_imgs // 16
            args['maxframes'] = 16
            args['ray_shoot_mode'] = 'image'
            args['keyfilter'] = cfg.test_keyfilter
        elif data_type == 'movement':
            if not 'skip' in args:
                args['skip'] = total_train_imgs // 64
            if not 'maxframes' in args:
                args['maxframes'] = 64   
        elif data_type == 'novelview':
            if cfg['type'] == 'train':
                if not 'skip' in args:
                    args['skip'] = total_train_imgs // 16
                args['maxframes'] = 16
                args['ray_shoot_mode'] = 'image'
                args['keyfilter'] = cfg.test_keyfilter
            else:
                pass #sub-sampling is already done in prepare_dataset.py  
        elif data_type == 'novelpose':
            if cfg['type'] == 'train':
                if not 'skip' in args:
                    args['skip'] = total_train_imgs // 16
                args['maxframes'] = 16
                args['ray_shoot_mode'] = 'image'
                args['keyfilter'] = cfg.test_keyfilter
            else:
                pass

    if data_type in ['freeview', 'tpose', 'tpose_pose_condition']:
        args['skip'] = cfg.render_skip

    dataset = _query_dataset(data_type)
    dataset = dataset(**args,**kwargs, dataset_name=dataset_name)
    return dataset


def _worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def create_dataloader(data_type='train', **kwargs):
    cfg_node = cfg[data_type]
    use_DDP = cfg['ddp']
    batch_size = cfg_node.batch_size
    shuffle = cfg_node.shuffle
    drop_last = cfg_node.drop_last

    dataset = create_dataset(data_type=data_type, **kwargs)

    if cfg.DEBUG or (data_type == 'novelpose_autoregressive' and cfg.rgb_history.length>0 and cfg.rgb_history.test_novelpose=='autoregressive'):
        num_workers = 0
        worker_init_fn = None
    else:
        num_workers = cfg.num_workers
        worker_init_fn = _worker_init_fn
    if use_DDP:
        sampler = DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                drop_last=drop_last,
                                                num_workers=num_workers,
                                                worker_init_fn=worker_init_fn,
                                                sampler=sampler)
    else:
        sampler = None
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                drop_last=drop_last,
                                                num_workers=num_workers,
                                                worker_init_fn=worker_init_fn)

    return data_loader, sampler
