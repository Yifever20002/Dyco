import os
import argparse

import torch

from third_parties.yacs import CfgNode as CN

# pylint: disable=redefined-outer-name

_C = CN()

# "resume" should be train options but we lift it up for cmd line convenience
_C.resume = False

# current iteration -- set a very large value for evaluation
_C.eval_iter = 10000000

# for rendering
_C.render_folder_name = ""
_C.ignore_non_rigid_motions = False
_C.render_skip = 1
_C.render_frames = 100
_C.eval_output_tag = ''

# for data loader
_C.DEBUG = (os.environ.get('DEBUG','False').lower()=='true')
_C.num_workers = 0 if _C.DEBUG else 4
_C.remove = False
_C.use_amp = False


def get_cfg_defaults():
    return _C.clone()


def parse_cfg(cfg):
    cfg.logdir = os.path.join('experiments', cfg.category, cfg.task, str(cfg.subject), cfg.experiment)


def determine_primary_secondary_gpus(cfg):
    print("------------------ GPU Configurations ------------------")
    cfg.n_gpus = torch.cuda.device_count()
    # print(torch.cuda.device_count())
    # assert False
    if cfg.n_gpus > 0:
        all_gpus = list(range(cfg.n_gpus))
        cfg.primary_gpus = [0]
        if cfg.n_gpus > 1:
            cfg.secondary_gpus = [g for g in all_gpus \
                                    if g not in cfg.primary_gpus]
        else:
            cfg.secondary_gpus = cfg.primary_gpus

    print(f"Primary GPUs: {cfg.primary_gpus}")
    print(f"Secondary GPUs: {cfg.secondary_gpus}")
    print("--------------------------------------------------------")


def save_cfg(cfg, file_path):
    print(f"Save the conifg as {file_path}")
    with open(file_path, 'w') as f:
        f.write(cfg.dump())

def make_cfg(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/default.yaml')
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg)
    if args.type!='skip':
        save_cfg_file = os.path.join(cfg.logdir, 'config.yaml')
        if os.path.isfile(save_cfg_file):
            print(f'Testing, Load config from {save_cfg_file}')
            cfg.merge_from_file(save_cfg_file)
            cfg.merge_from_list(args.opts) # overwrite again!
    determine_primary_secondary_gpus(cfg)
    cfg.test.type = args.type

        
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", required=True, type=str)
parser.add_argument("--type", default="skip", type=str)
parser.add_argument("--pose_id", default="313", type=str)
parser.add_argument("--test_num", default=-1, type=int)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

cfg = make_cfg(args)

