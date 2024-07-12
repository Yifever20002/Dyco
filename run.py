import os

import torch, math
import numpy as np
from tqdm import tqdm
from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image
from core.utils.metrics_util import MetricsWriter
from configs import cfg, args
from collections import defaultdict
from utils import custom_print
EXCLUDE_KEYS_TO_GPU = ['frame_name',
                       'img_width', 'img_height', 'ray_mask','img_name']

# import DDP #
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from utils import get_local_rank, init_env, is_master, get_rank, get_world_size
import torch.multiprocessing as mp


def load_network():
    model = create_network()
    latest = max((f for f in os.listdir(cfg.logdir) if f.startswith('iter_') and f.endswith('.tar')),
                   key=lambda x: int(x.split('_')[1].split('.')[0]), default=None)
    ckpt_path = os.path.join(cfg.logdir, latest)
    ckpt = torch.load(ckpt_path, map_location='cuda:0')

    model.load_state_dict(ckpt['network'], strict=False)
    custom_print('load network from ', ckpt_path)
    if cfg['ddp']:
        return model.cuda()
    return model.cuda().deploy_mlps_to_secondary_gpus()


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))

def unpack_weight_map(weight_vals, ray_mask, width, height, weight_mask=None):
    weight_map = np.zeros((height * width, weight_vals.shape[-1]), dtype='float32')
    if weight_mask is not None:
        weight_vals[weight_mask==False] = 0
    weight_map[ray_mask,:] = weight_vals #(N,
    return weight_map.reshape((height, width,-1))

def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None):
    
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image  = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image


def _freeview(
        data_type='freeview',
        folder_name=None, render_folder_name='freeview', **kwargs):
    cfg.perturb = 0.

    model = load_network()
    if cfg['ddp']:
        model = DDP(model, device_ids=cfg['device_ids'], output_device=get_local_rank(), find_unused_parameters=False)
    test_loader = create_dataloader(data_type, **kwargs)
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net+cfg.eval_output_tag),
        exp_name=render_folder_name)

    model.eval()
    step = 0

    for batch in tqdm(test_loader):
        step += 1
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU)

        with torch.no_grad():
            net_output = model(**data, 
                               iter_val=cfg.eval_iter)
        
        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']
        rgb = net_output['rgb']
        alpha = net_output['alpha']
        depth = net_output['depth']
        weights_on_ray, xyz_on_ray, rgb_on_ray = net_output['weights_on_rays'],net_output['xyz_on_rays'],net_output['rgb_on_rays']
        img_name = batch.get('img_name', None)     
        target_rgbs = batch.get('target_rgbs', None)
        raw_rgbs = batch.get('raw_rgbs', None)
        rgb_img, alpha_img, _ = unpack_to_image(
            width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
            rgb.data.cpu().numpy(),                    
            alpha.data.cpu().numpy())
        #depth_img = unpack_alpha_map(alpha_vals=depth, ray_mask=ray_mask, width=width, height=height)
        imgs = [rgb_img]
        if cfg.show_truth and target_rgbs is not None:
            raw_rgbs = to_8b_image(raw_rgbs.numpy())
            imgs.append(raw_rgbs)
        if cfg.show_alpha:
            imgs.append(alpha_img)
        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=img_name)
        if cfg.test.save_3d:
            weight_mask = (weights_on_ray.max(axis=1)[0]>cfg.test.weight_threshold) #R,N -> R,
            xyzs = torch.sum(xyz_on_ray[weight_mask]*weights_on_ray[weight_mask][...,None],axis=1) #R,N,3*R,N,1 ->R,N
            rgbs = torch.sum(rgb_on_ray[weight_mask]*weights_on_ray[weight_mask][...,None],axis=1) #R,N,3*R,N,1 ->R,N
            #cnl_xyz, cnl_rgb = xyz_on_ray[weight_mask].data.cpu().numpy(), rgb_on_ray[weight_mask].data.cpu().numpy()
            writer.append_cnl_3d(xyzs.data.cpu().numpy(), rgbs.data.cpu().numpy(), obj_name=str(step)+'-cnl')
        #metrics_writer.append(name=img_name, pred=rgb_img, target=raw_rgbs)

    writer.finalize()


def run_freeview():
    _freeview(
        data_type='freeview',
        render_folder_name=f"freeview_{cfg.freeview.frame_idx}" \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_tpose():
    cfg.ignore_non_rigid_motions = True
    _freeview(
        data_type='tpose',
        render_folder_name='tpose' \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_tpose_pose_condition():
    cfg.ignore_non_rigid_motions = True
    import os
    if int(os.environ.get('FORCE_NON_RIGID_MOTIONS',0))==1:
        cfg.ignore_non_rigid_motions = False
        _freeview(
            data_type='tpose_pose_condition',
            render_folder_name='tpose_pose_condition_w-delta' \
                if not cfg.render_folder_name else cfg.render_folder_name)
    else:
        _freeview(
            data_type='tpose_pose_condition',
            render_folder_name='tpose_pose_condition' \
                if not cfg.render_folder_name else cfg.render_folder_name)
 

def run_novelview():
    cfg.show_truth = True
    run_movement(render_folder_name='novelview')

def run_novelview_all():
    cfg.show_truth = True
    run_movement(render_folder_name='novelview_all')

def run_novelpose_autoregressive():
    cfg.show_truth = True
    if cfg.rgb_history.length>0:
        cfg.eval_output_tag = '-'+cfg.rgb_history.test_novelpose
        if cfg.rgb_history.test_novelpose=='autoregressive':
            cfg.rgb_history.precompute = 'empty' 
            cfg.test.save_depth = True
        run_movement(render_folder_name='novelpose_autoregressive')
    else:
        run_movement(render_folder_name='novelpose_autoregressive')

def run_novelpose():
    cfg.show_truth = True
    if cfg.rgb_history.length>0:
        if cfg.rgb_history.test_novelpose=='autoregressive':
            cfg.rgb_history.precompute = 'empty'     
            cfg.novelpose.dataset += '_autoregressive'
            cfg.test.save_depth = True
        elif cfg.rgb_history.test_novelpose=='oracle':
            cfg.eval_output_tag = '-'+cfg.rgb_history.test_novelpose
        else:
            raise ValueError
        run_movement(render_folder_name='novelpose')
    else:
        run_movement(render_folder_name='novelpose')

def run_novelpose_comb():
    run_movement(render_folder_name='novelpose_comb')

def run_progress():
    run_movement(render_folder_name='progress')
    
def run_stopwrun():
    run_movement(render_folder_name='stopwrun')

def run_novelview_res():
    run_movement(render_folder_name='novelview_res')

def run_movement(render_folder_name='movement'):
    cfg.perturb = 0.
    cfg.show_truth = True
    model = load_network()
    if cfg['ddp']:
        model = DDP(model, device_ids=cfg['device_ids'], output_device=get_local_rank(), find_unused_parameters=False)
    if cfg['ddp']:
        if get_local_rank() == 0:
            writer = ImageWriter(
                output_dir=os.path.join(cfg.logdir, cfg.load_net+cfg.eval_output_tag),
                exp_name=render_folder_name)
            cfg.rgb_history.novelpose_image_dir = writer.image_dir 
            cfg.rgb_history.novelpose_depth_dir = writer.depth_dir
    else:
        writer = ImageWriter(
            output_dir=os.path.join(cfg.logdir, cfg.load_net+cfg.eval_output_tag),
            exp_name=render_folder_name)
        cfg.rgb_history.novelpose_image_dir = writer.image_dir 
        cfg.rgb_history.novelpose_depth_dir = writer.depth_dir


    test_loader, _ = create_dataloader(render_folder_name)
    if cfg['ddp']:
        if get_local_rank() == 0:
            metrics_writer = MetricsWriter(
                    output_dir=os.path.join(cfg.logdir, cfg.load_net+cfg.eval_output_tag), 
                    exp_name=render_folder_name,
                    dataset=cfg[render_folder_name].dataset)
    else:
        metrics_writer = MetricsWriter(
            output_dir=os.path.join(cfg.logdir, cfg.load_net+cfg.eval_output_tag), 
            exp_name=render_folder_name,
            dataset=cfg[render_folder_name].dataset)

    model.eval()
    if os.environ.get('RETURN_POSE','False').lower()=='true':
        pose_refine_output = {}

    for idx, batch in enumerate(tqdm(test_loader)):
        if args.test_num!=-1 and idx>=args.test_num:
            break
        for k, v in batch.items():
            if k=='frame_name_history':
                batch[k] = [[v2[0] for v2 in v1] for v1 in v]
            else:
                batch[k] = v[0]
        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs','frame_name_history'])

        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter)

        rgbs = [net_output['rgb']]
        alphas = [net_output['alpha']]
        depths = [net_output['depth']]
        offsets = [net_output['offsets']]
        backward_motion_weights = net_output['backward_motion_weights']
        weights_on_rays, xyz_on_rays, rgb_on_rays = [net_output['weights_on_rays']],[net_output['xyz_on_rays']],[net_output['rgb_on_rays']]
        cnl_xyzs, cnl_rgbs, cnl_weights = [net_output['cnl_xyz']],[net_output['cnl_rgb']], [net_output['cnl_weight']]
        img_names = [None] 

        for hid,(rgb, alpha, depth, cnl_xyz, cnl_rgb, cnl_weight, weights_on_ray, xyz_on_ray, rgb_on_ray, offset_on_ray, img_name) in \
                enumerate(zip(rgbs, alphas, depths, cnl_xyzs, cnl_rgbs, cnl_weights, weights_on_rays, xyz_on_rays, rgb_on_rays, offsets, img_names)):
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rgb_img, alpha_img, truth_img = \
                unpack_to_image(
                    width, height, ray_mask, np.array(cfg.bgcolor)/255.,
                    rgb.data.cpu().numpy(),
                    alpha.data.cpu().numpy(),
                    batch['target_rgbs'])

            imgs = [rgb_img]
            if cfg.show_truth:
                imgs.append(truth_img)
            if cfg.show_alpha:
                imgs.append(alpha_img)
            
            if idx%cfg.eval_step==0:
                if cfg['ddp']:
                    gather_rgb = gather_together(rgb_img)
                    gather_alpha = gather_together(alpha_img)
                    gather_truth = gather_together(truth_img)
                    gather_name = gather_together(batch['frame_name'])
                    if get_local_rank() == 0:
                        for i in range(len(gather_rgb)):
                            metrics_writer.append(name=gather_name[i].replace('/','-'), pred=gather_rgb[i], target=gather_truth[i], mask=None)
                            imgs = [gather_rgb[i]]
                            if cfg.show_truth:
                                imgs.append(gather_truth[i])
                            if cfg.show_alpha:
                                imgs.append(gather_alpha[i])
                            img_out = np.concatenate(imgs, axis=1)
                            writer.append(img_out, img_name=gather_name[i].replace('/','-'))
                else:
                    metrics_writer.append(name=batch['frame_name'], pred=rgb_img, target=truth_img, mask=None)
                    img_out = np.concatenate(imgs, axis=1)
                    writer.append(img_out, img_name=batch['frame_name'].replace('/','-'))
            
            if cfg.test.save_depth:
                depth_img = unpack_alpha_map(alpha_vals=depth.data.cpu().numpy(), ray_mask=ray_mask, width=width, height=height)
                writer.append_depth(depth_img, img_name=batch['frame_name'])


            if cfg.test.save_3d_together:
                #use ray_mask!
                rgb_on_image = batch['target_rgbs'].to(weights_on_ray.device)
                weighted_xyz = torch.sum(weights_on_ray[...,None]*xyz_on_ray, axis=1)
                weight_max = torch.max(weights_on_ray, axis=-1)[0][...,None]
                lbs = torch.sum(weights_on_ray[...,None]*backward_motion_weights, axis=1)
                lbs_argmax = torch.argmax(lbs, axis=1)[...,None] #N
                pos_on_image = (ray_mask.view((height, width))).nonzero().to(weights_on_ray.device)
                save_mask = (torch.max(weights_on_ray,axis=1)[0])>cfg.test.weight_threshold
                save_mask = save_mask.to(weights_on_ray.device)
                writer.append_3d_together(
                    name=batch['frame_name'],
                    data=torch.cat([weighted_xyz[save_mask], 
                                    rgb_on_image[save_mask], 
                                    weight_max[save_mask], 
                                    pos_on_image[save_mask], 
                                    lbs_argmax[save_mask]], axis=1)) #N,(3+3+1)


            if cfg.test.save_3d:
                pos_on_image = (ray_mask.view((height, width))).nonzero() #N_rays, 2
                rgb_on_image = batch['target_rgbs'] #N_rays, 3
                writer.save_pkl({'weights_on_rays':weights_on_ray.data.cpu().numpy(), 
                             'rgb_on_rays':rgb_on_ray.data.cpu().numpy(), 
                             'xyz_on_rays':xyz_on_ray.data.cpu().numpy(),
                             'rgb_on_image':rgb_on_image.data.cpu().numpy(),
                             'pos_on_image':pos_on_image.data.cpu().numpy(),
                             'offset_on_rays':offset_on_ray.data.cpu().numpy(),
                             'cnl_xyz':cnl_xyz.data.cpu().numpy()}, name=batch['frame_name'].replace('/','-')+'-rays.pkl')

    if cfg['ddp']:
        if get_local_rank() == 0:
            metrics_writer.finalize()
            writer.finalize()
    else:
        metrics_writer.finalize()
        writer.finalize()

    
    if os.environ.get('RETURN_POSE','False').lower()=='true':
        import pickle
        with open(os.path.join(metrics_writer.output_dir, f'{metrics_writer.exp_name}-pose_refine_output.pkl'),'wb') as f:
            pickle.dump(pose_refine_output, f)

def gather_together(data):
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data

        
if __name__ == '__main__':
    cfg['type'] = args.type
    init_env(cfg)
    globals()[f'run_{args.type}']()
