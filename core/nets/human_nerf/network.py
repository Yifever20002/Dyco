import torch, os
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import MotionBasisComputer
from core.utils.transformation_util import axis_angle_to_matrix, axis_angle_to_quaternion
from core.nets.human_nerf.component_factory import \
    load_positional_embedder, \
    load_canonical_mlp, \
    load_mweight_vol_decoder, \
    load_pose_decoder, \
    load_non_rigid_motion_mlp, \
    load_non_rigid_motion_transformer_encoder, \
    load_vocab_embedder
from core.nets.human_nerf.rgb_feature import RGB_FeatureIndexer
from configs import cfg

import tinycudann as tcnn
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable
import itertools

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(
                                        total_bones=cfg.total_bones)

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
            embedding_size=cfg.mweight_volume.embedding_size,
            volume_size=cfg.mweight_volume.volume_size,
            total_bones=cfg.total_bones
        )

        # non-rigid motion st positional encoding
        self.get_non_rigid_embedder = \
            load_positional_embedder(cfg.non_rigid_embedder.module)

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = \
            self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, 
                                        cfg.non_rigid_motion_mlp.i_embed)
        if cfg.non_rigid_motion_model in ['mlp','mlp_SA']:
            cfg_ = cfg.non_rigid_motion_mlp \
                 if cfg.non_rigid_motion_model == 'mlp' else cfg.non_rigid_motion_mlp_sa
            sa = {} if cfg.non_rigid_motion_model == 'mlp'  else cfg_.sa
            self.non_rigid_mlp = \
                load_non_rigid_motion_mlp(cfg_.module)(
                    pos_embed_size=non_rigid_pos_embed_size,
                    mlp_width=cfg_.mlp_width,
                    mlp_depth=cfg_.mlp_depth,
                    skips=cfg_.skips,
                    last_linear_scale=cfg_.last_linear_scale,
                    mlp_depth_plus=cfg_.mlp_depth_plus, **sa)
        elif cfg.non_rigid_motion_model == 'transformer_encoder':
            self.non_rigid_mlp = \
                load_non_rigid_motion_transformer_encoder(cfg.non_rigid_motion_transformer_encoder.module)(
                    query_input_dim=non_rigid_pos_embed_size) # already in cfg.non_rigid_motion_transformer_encoder
        elif cfg.non_rigid_motion_model == 'TStransformer_encoder':
            self.non_rigid_mlp = \
                load_non_rigid_motion_transformer_encoder(cfg.non_rigid_motion_TStransformer_encoder.module)(
                    query_input_dim=non_rigid_pos_embed_size) # already in cfg.non_rigid_motion_transformer_encoder
        if not cfg['ddp']:              
            self.non_rigid_mlp = \
                nn.DataParallel(
                    self.non_rigid_mlp,
                    device_ids=cfg.secondary_gpus,
                    output_device=cfg.secondary_gpus[0])

        # canonical positional encoding
        get_embedder = load_positional_embedder(cfg.embedder.module)
        cnl_pos_embed_fn, cnl_pos_embed_size = \
            get_embedder(cfg.canonical_mlp.multires, 
                         cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn
        
        if cfg.canonical_mlp.view_dir:
            if cfg.canonical_mlp.view_embed == 'mlp':
                get_embedder_dir = load_positional_embedder(cfg.embedder.module)
                self.dir_embed_fn, cnl_dir_embed_size = \
                    get_embedder_dir(cfg.canonical_mlp.multires_dir, 
                                cfg.canonical_mlp.i_embed)
            elif cfg.canonical_mlp.view_embed == 'vocab':
                get_embedder_dir = load_vocab_embedder(cfg.vocab_embedder.module)
                self.dir_embed_fn, cnl_dir_embed_size = \
                    get_embedder_dir(cfg.canonical_mlp.view_vocab_n, 
                                    cfg.canonical_mlp.view_vocab_dim)
            else:
                raise ValueError
        else:
            self.dir_embed_fn, cnl_dir_embed_size = None, -1

        if cfg.canonical_mlp.triplane:
            # Init planes
            self.grids = nn.ModuleList()
            self.multiscale_res_multipliers = [1, 2, 4, 8]
            self.feature_dim = 0
            self.grid_config = [{
                'grid_dimensions': 2,
                'input_coordinate_dim': 3,
                'output_coordinate_dim': 32,
                'resolution': [64, 64, 64]
                }]
            for res in self.multiscale_res_multipliers:
                # initialize coordinate grid
                config = self.grid_config[0].copy()
                # Resolution fix: multi-res only on spatial planes
                config["resolution"] = [
                    r * res for r in config["resolution"][:3]
                ] + config["resolution"][3:]
                gp = init_grid_param(
                    grid_nd=config["grid_dimensions"],
                    in_dim=config["input_coordinate_dim"],
                    out_dim=config["output_coordinate_dim"],
                    reso=config["resolution"],
                )

                # shape[1] is out-dim - Concatenate over feature len for each scale
                self.feature_dim += gp[-1].shape[1]
                self.grids.append(gp)

        # canonical mlp 
        skips = [4]
        self.cnl_mlp = \
            load_canonical_mlp(cfg.canonical_mlp.module)(
                pos_embed_dim=cnl_pos_embed_size, 
                mlp_depth=cfg.canonical_mlp.mlp_depth, 
                mlp_width=cfg.canonical_mlp.mlp_width,
                view_dir=cfg.canonical_mlp.view_dir, 
                input_ch_dir=cnl_dir_embed_size, 
                pose_color=cfg.canonical_mlp.pose_color,
                pose_ch=cfg.canonical_mlp.pose_ch,
                skips=skips,
                last_linear_scale=cfg.canonical_mlp.last_linear_scale,
                mlp_depth_plus=cfg.canonical_mlp.mlp_depth_plus)
        if not cfg['ddp']:
            self.cnl_mlp = \
                nn.DataParallel(
                    self.cnl_mlp,
                    device_ids=cfg.secondary_gpus,
                    output_device=cfg.primary_gpus[0])

        # pose decoder MLP
        if cfg.pose_decoder_off == False:
            self.pose_decoder = \
                load_pose_decoder(cfg.pose_decoder.module)(
                    embedding_size=cfg.pose_decoder.embedding_size,
                    mlp_width=cfg.pose_decoder.mlp_width,
                    mlp_depth=cfg.pose_decoder.mlp_depth)
          
        if cfg.rgb_history.length > 0:
            self.rgb_feature_indexer = RGB_FeatureIndexer(**cfg.rgb_history,)
            if not cfg['ddp']:
                self.rgb_feature_indexer = \
                    nn.DataParallel(
                        self.rgb_feature_indexer,
                        device_ids=cfg.secondary_gpus,
                        output_device=cfg.primary_gpus[0])
        else:
            self.rgb_feature_indexer = None




    def deploy_mlps_to_secondary_gpus(self):
        self.cnl_mlp = self.cnl_mlp.to(cfg.secondary_gpus[0])
        if self.non_rigid_mlp:
            self.non_rigid_mlp = self.non_rigid_mlp.to(cfg.secondary_gpus[0])

        return self


    def _query_mlp(
            self,
            pos_xyz,
            pos_embed_fn, 
            non_rigid_pos_embed_fn,
            dir_xyz, dir_idx, 
            dir_embed_fn, 
            backward_motion_weights, iter_val, aabb, **kwargs):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        weights_flat = torch.reshape(backward_motion_weights, [-1, backward_motion_weights.shape[-1]])
        if cfg.canonical_mlp.view_embed == 'mlp':
            dir_flat = torch.reshape(dir_xyz, [-1, dir_xyz.shape[-1]])
        elif cfg.canonical_mlp.view_embed == 'vocab':
            dir_flat = torch.reshape(dir_idx, [-1,]) #N,
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        pos_embed_fn=pos_embed_fn,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                        dir_flat=dir_flat, 
                        dir_embed_fn=dir_embed_fn,
                        chunk=chunk, weights_flat=weights_flat, iter_val=iter_val, aabb=aabb,
                        **kwargs)

        output = {}

        raws_flat = result['raws']
        xyzs_flat = result['xyzs'] #batch, 3 or [(batch,3), (batch,3)]
        offsets_flat = result['offsets']
        output['raws'] = torch.reshape(
                            raws_flat, 
                            list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]]) 
        output['xyzs'] = torch.reshape(
                            xyzs_flat, 
                            list(pos_xyz.shape[:-1]) + [xyzs_flat.shape[-1]])                     
        output['offsets'] = torch.reshape(
                            offsets_flat, 
                            list(pos_xyz.shape[:-1]) + [offsets_flat.shape[-1]])          

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1:]
        return input_data.expand((total_elem, )+input_size)
    


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            dir_flat, 
            dir_embed_fn,
            chunk, weights_flat, iter_val, rgb_history_input={},
            non_rigid_motion_pose_condition=None, 
            canonical_pose_condition=None,
            non_rigid_motion_posedelta_condition=None, 
            canonical_posedelta_condition=None,
            depth_history=None,
            aabb=None,
             **kwargs):

        raws = []
        xyzs = []
        offsets = []
        output = {}
        # iterate ray samples by trunks
        if not cfg['ddp']:
            if non_rigid_motion_pose_condition is not None:
                non_rigid_motion_pose_condition = self._expand_input(non_rigid_motion_pose_condition[None,...], len(cfg.secondary_gpus))
            if canonical_pose_condition is not None:
                canonical_pose_condition = self._expand_input(canonical_pose_condition[None,...], len(cfg.secondary_gpus))
            if non_rigid_motion_posedelta_condition is not None:
                non_rigid_motion_posedelta_condition = self._expand_input(non_rigid_motion_posedelta_condition[None,...], len(cfg.secondary_gpus))
            if canonical_posedelta_condition is not None:
                canonical_posedelta_condition = self._expand_input(canonical_posedelta_condition[None,...], len(cfg.secondary_gpus))
            if rgb_history_input != {}:
                rgb_history_input = {k:self._expand_input(v[None,...], len(cfg.secondary_gpus)) for k,v in rgb_history_input.items()}
        
        else:
            if non_rigid_motion_pose_condition is not None:
                non_rigid_motion_pose_condition = self._expand_input(non_rigid_motion_pose_condition[None,...], 1)
            if canonical_pose_condition is not None:
                canonical_pose_condition = self._expand_input(canonical_pose_condition[None,...], 1)
            if non_rigid_motion_posedelta_condition is not None:
                non_rigid_motion_posedelta_condition = self._expand_input(non_rigid_motion_posedelta_condition[None,...], 1)
            if canonical_posedelta_condition is not None:
                canonical_posedelta_condition = self._expand_input(canonical_posedelta_condition[None,...], 1)
            if rgb_history_input != {}:
                rgb_history_input = {k:self._expand_input(v[None,...], 1) for k,v in rgb_history_input.items()}

        for i in range(0, pos_flat.shape[0], chunk):
            # print('pos_flat', pos_flat.shape[0], 'chunk',chunk)
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start


            xyz, dir_ = pos_flat[start:end], dir_flat[start:end]
            weights = weights_flat[start:end]

            if not cfg.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)

                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    pose_condition=non_rigid_motion_pose_condition, #gpu, (N), dim
                    posedelta_condition=non_rigid_motion_posedelta_condition, #gpu, (N), dim
                    weights=weights)

                xyz = result['xyz'] #B, 3 or list
                ofs = result['offsets']
            else:
                ofs = torch.zeros_like(xyz)
            if cfg.canonical_mlp.view_dir:
                dir_embed = dir_embed_fn(dir_) #vocab: anyshape
            else:
                dir_embed = None
                     
            xyz_embedded = pos_embed_fn(xyz) #B*n_head (if argmin), 3*2*10
            xyzs.append(xyz)
            offsets.append(ofs) #N,

            if self.rgb_feature_indexer is not None:
                rgb_condition, visible = self.rgb_feature_indexer(
                    cnl_pts=xyz.detach(), forward_motion_weights=weights.detach(), 
                    depth_history=depth_history, #(1,T,V,H,W)
                    **rgb_history_input) #forward_motion_scale_Rs_history, forward_motion_Ts_history, w2cs_history, rgb_history
            else:
                rgb_condition = None
                visible = None

            features = None
            if cfg.canonical_mlp.triplane:
                xyz = normalize_aabb(xyz, aabb)
                features = interpolate_ms_features(
                    xyz, ms_grids=self.grids,  # noqa
                    grid_dimensions=self.grid_config[0]["grid_dimensions"],
                    concat_features=True, num_levels=None)

            raws += [self.cnl_mlp(
                        features_xyz=features, pos_xyz=xyz, pos_embed=xyz_embedded, dir_embed=dir_embed, 
                        pose_condition=canonical_pose_condition,
                        posedelta_condition=canonical_posedelta_condition,
                        rgb_condition=rgb_condition, visible=visible, 
                        weights=weights,iter_val=iter_val,)] #N*num_head, 4
        
        if cfg['ddp']:
            output['raws'] = torch.cat(raws, dim=0).to(torch.cuda.current_device()) #N*num_head, 4
            output['xyzs'] = torch.cat(xyzs, dim=0).to(torch.cuda.current_device())
            output['offsets'] = torch.cat(offsets, dim=0).to(torch.cuda.current_device())
        else:
            output['raws'] = torch.cat(raws, dim=0).to(cfg.primary_gpus[0]) #N*num_head, 4
            output['xyzs'] = torch.cat(xyzs, dim=0).to(cfg.primary_gpus[0])
            output['offsets'] = torch.cat(offsets, dim=0).to(cfg.primary_gpus[0])
        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        multi_outputs = False
        for iii,i in enumerate(range(0, rays_flat.shape[0], cfg.chunk)):
            # print(iii, rays_flat.shape[0], cfg.chunk)
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
            for k in ret: #rgb, depth
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k]) 
        if multi_outputs:
            all_ret = {k : [torch.cat(x, 0) for x in all_ret[k]] for k in all_ret} #'rgb':[tensor1-for head0, tensor2]
        else:
            all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret} 
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, xyz, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha0 = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        alpha = alpha0 * raw_mask[:, :, 0] #foreground probability

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1] #[N_rays, n_sample]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, n_samples, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.
        
        #xyz [N_rays, n_sample, 3]
        
        
        weights_max, indices = weights.max(dim=1) #N_rays
        indices = indices[:,None,None] #N_rays, 1, 1
        cnl_xyz = torch.gather(xyz, dim=1, index=indices.tile([1,1,xyz.shape[-1]]))
        cnl_rgb = torch.gather(rgb, dim=1, index=indices.tile([1,1,rgb.shape[-1]]))

        fg_mask_only = torch.max(raw_mask[:,:,0], dim=1)[0]
        alpha_only = torch.max(alpha0, dim=1)[0]
        return rgb_map, acc_map, weights, depth_map, cnl_xyz.squeeze(1), cnl_rgb.squeeze(1), weights_max, rgb, fg_mask_only, alpha_only
        


    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_scale_Rs, 
            motion_Ts, 
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 

        weights_list = []
        for i in range(motion_weights.size(0)): #iteratve over joints
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0 
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], 
                                    grid=pos[None, None, None, :, :],         
                                    padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] 
            weights_list.append(weights) 
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:2]+[3])
        backwarp_motion_weights = \
            backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        if 'backward_motion_weights' in output_list:
            results['backward_motion_weights'] = backwarp_motion_weights
        return results


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d, rays_d_camera = ray_batch[:,0:3], ray_batch[:,3:6], ray_batch[:,6:9]
        bounds = torch.reshape(ray_batch[...,9:11], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, rays_d_camera, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            dir_embed_fn,
            bgcolor=None, 
            dir_idx=None, iter_val=1e7,
            **kwargs):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, rays_d_camera, near, far = self._unpack_ray_batch(ray_batch)
        self.cnl_bbox_scale_xyz, self.cnl_bbox_min_xyz = cnl_bbox_scale_xyz, cnl_bbox_min_xyz
        self.aabb = [self.cnl_bbox_min_xyz, self.cnl_bbox_scale_xyz]
        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] #6144, 128, 3
        if os.environ.get('TEST_DIR', '') != '':
            dir_xyz = torch.nn.functional.normalize(_['rays_d_'].float())[:,None,:] # N,1,3
        else:
            if cfg.canonical_mlp.view_dir_camera_only==True:
                dir_xyz = torch.nn.functional.normalize(rays_d_camera)[:,None,:] # N,1,3
            else:
                dir_xyz = torch.nn.functional.normalize(rays_d)[:,None,:] # N,1,3
        dir_xyz = torch.tile(dir_xyz, [1,pts.shape[1],1])
        if dir_idx is None:
            dir_idx = torch.zeros([dir_xyz.shape[0]*dir_xyz.shape[1], 1], dtype=torch.long, device=dir_xyz.device)
        else:
            dir_idx = torch.tile(dir_idx[:, None], [dir_xyz.shape[0],pts.shape[1]]) #(1,1)->(N-ray, N-point)
        mv_output = self._sample_motion_fields(
                            pts=pts,
                            motion_scale_Rs=motion_scale_Rs[0], 
                            motion_Ts=motion_Ts[0], 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask', 'backward_motion_weights'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']
        backward_motion_weights = mv_output['backward_motion_weights']

        outputs = {}

        query_result = self._query_mlp(
                                pos_xyz=cnl_pts, 
                                pos_embed_fn=pos_embed_fn,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                                dir_embed_fn=dir_embed_fn,
                                dir_xyz=dir_xyz, dir_idx=dir_idx, 
                                backward_motion_weights=backward_motion_weights,
                                iter_val=iter_val,
                                aabb=self.aabb, **kwargs) #kwargs, pose_condition, rgb_history_input
        raw = query_result['raws']
        xyz = query_result['xyzs']
                  
        rgb_map, acc_map, weights, depth_map, cnl_xyz, cnl_rgb, cnl_weight, rgb_on_rays, fg_mask_only, alpha_only = \
            self._raw2outputs(raw, pts_mask, z_vals, rays_d, xyz, bgcolor) #[N_rays, 3]
            #multi_outputs = False

        outputs.update({'rgb' : rgb_map,  
                'alpha' : acc_map, 
                'depth': depth_map,
                'weights_on_rays': weights,
                'xyz_on_rays': xyz, 'rgb_on_rays': rgb_on_rays, 
                'cnl_xyz':cnl_xyz, 'cnl_rgb':cnl_rgb, 'cnl_weight':cnl_weight,
                'backward_motion_weights': backward_motion_weights, #unnormalized
                'offsets': query_result['offsets'], 'fg_mask_only': fg_mask_only, 'alpha_only':alpha_only})#, multi_outputs
        return outputs


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    def correspondence_forward_searching(self, pts, forward_motion_weights, dst_Rs, dst_Ts):
        orig_shape = list(pts.shape) #N_raysxN_samples,3 or N,3
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]
        forward_motion_weights = forward_motion_weights.reshape(-1, forward_motion_weights.shape[-1])
        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs, 
                                            dst_Ts=dst_Ts, 
                                            cnl_gtfms=self.cnl_gtfms.expand((dst_Rs.shape[0],-1,-1,-1)).contiguous()) #[N,24,3,3]
        motion_scale_Rs_forward = motion_scale_Rs.transpose(2,3) #!!! Important!!
        motion_Ts_forward = -1*torch.einsum('bnij,bnj->bni', motion_scale_Rs_forward, motion_Ts) #(B,N,3) #!!! Important!!
        total_bases = forward_motion_weights.shape[-1]
        forward_motion_weights_sum = torch.sum(forward_motion_weights, 
                                                dim=-1, keepdim=True) #(B,1)
        weighted_motion_fields = []
        for i in range(total_bases):
            #pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            #pos = torch.matmul(motion_scale_Rs[i, :, :].T, pts.T).T  - torch.matmul(motion_scale_Rs[i, :, :].T,motion_Ts[i, :].T).T #inverse (B,3)
            pos = torch.einsum('nij,bj->bni', motion_scale_Rs_forward[:,i], pts)+motion_Ts_forward[:,i]  #N,3,3, (b,3)-> (b,N,i)
            weighted_pos = forward_motion_weights[:,None,i:i+1] * pos #(b,1,1) (b,N,3)->(b,N,3)
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / forward_motion_weights_sum.clamp(min=0.0001)[:,None,:] #(b,N,3)/(B,1,1)
        # fg_likelihood_mask = forward_motion_weights_sum
        #(B,N,3)
        x_skel = x_skel.reshape(orig_shape[:-1]+[-1,3])
        # fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])
        return x_skel
    
    def forward(self,
                rays, 
                dst_Rs, dst_Ts, cnl_gtfms,
                motion_weights_priors,
                dst_posevec=None,
                near=None, far=None,
                iter_val=1e7,
                frame_id=None, 
                dst_posevec_history=None,
                dst_Rs_history=None, dst_Ts_history=None,
                **kwargs):
        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        self.cnl_gtfms = cnl_gtfms
        motion_weights_priors=motion_weights_priors[None, ...]

        # correct body pose
        if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0) and cfg.pose_decoder_off==False:
            if dst_Rs_history is not None:
                dst_posevec = torch.cat([dst_posevec, dst_posevec_history], dim=0)
                dst_Rs =  torch.cat([dst_Rs, dst_Rs_history], dim=0)
                dst_Ts =  torch.cat([dst_Ts, dst_Ts_history], dim=0)
            pose_out = self.pose_decoder(dst_posevec)
            refined_Rs = pose_out['Rs']
            refined_rvec = pose_out['rvec']
            refined_Ts = pose_out.get('Ts', None)
            
            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            # pose_refine_output = {
            #     'delta_r': refined_rvec.cpu().numpy(), 
            #     'R0': dst_Rs_no_root.cpu().numpy(),
            #     'r0': dst_posevec.cpu().numpy()}
            dst_Rs_no_root = dst_Rs_no_root.expand((refined_Rs.shape[0],-1,-1,-1))
            dst_Rs_no_root = self._multiply_corrected_Rs(
                                        dst_Rs_no_root, 
                                        refined_Rs)
            #pose_refine_output['R1'] = dst_Rs_no_root.cpu().numpy()
            dst_Rs = torch.cat(
                [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

            if refined_Ts is not None:
                dst_Ts = dst_Ts + refined_Ts
            
            if dst_Rs_history is not None:
                dst_Rs_history = dst_Rs[1:].detach()
                dst_Ts_history = dst_Ts[1:].detach()
            dst_Rs, dst_Ts = dst_Rs[0:1], dst_Ts[0:1]

        if cfg.rgb_history.length > 0: #compute forward_motion_scale_Rs, forward_motion_Ts,
            motion_scale_Rs_history, motion_Ts_history = self._get_motion_base(
                                                dst_Rs=dst_Rs_history, 
                                                dst_Ts=dst_Ts_history, 
                                                cnl_gtfms=self.cnl_gtfms.expand((dst_Rs_history.shape[0],-1,-1,-1)).contiguous()) #[T,24,3,3]
            forward_motion_scale_Rs_history = motion_scale_Rs_history.transpose(2,3) #!!! Important!!
            forward_motion_Ts_history = -1*torch.einsum('bnij,bnj->bni', forward_motion_scale_Rs_history, motion_Ts_history) #(T,24,3) #!!! Important!!
            kwargs['rgb_history_input'] = {
                'forward_motion_scale_Rs_history':forward_motion_scale_Rs_history, 
                'forward_motion_Ts_history':forward_motion_Ts_history, 
                'w2c_history': kwargs.pop('w2c_history'), 
                'rgb_history': kwargs.pop('rgb_history'),
            }
        else:
            kwargs['rgb_history_input'] = {}

        non_rigid_pos_embed_fn, _ = \
            self.get_non_rigid_embedder(
                multires=cfg.non_rigid_motion_mlp.multires,                         
                is_identity=cfg.non_rigid_motion_mlp.i_embed,
                iter_val=iter_val,)
        

        # non_rigid_mlp_input, non_rigid_mlp_input_dict = [], {}
        # if cfg.non_rigid_motion_mlp.pose_input and not cfg.ignore_non_rigid_motions:
            # if pose_condition is not None:
            #     dst_posevec = pose_condition[None,...]
            # else:
            #     if cfg.posevec.type == 'axis_angle':
            #         pass 
            #     elif cfg.posevec.type == 'matrix':
            #         dst_posevec = dst_posevec.reshape(list(dst_posevec.shape[:-1])+[dst_posevec.shape[-1]//3,3]) #*,N,3
            #         rest_matrix = axis_angle_to_matrix(torch.zeros_like(dst_posevec))  #I
            #         pose_matrix = axis_angle_to_matrix(dst_posevec)
            #         dst_posevec = rest_matrix-pose_matrix #*,N,3,3 # so that I -> 0
            #         dst_posevec = dst_posevec.reshape(list(dst_posevec.shape[:-3])+[-1]) #*,N*9
            #     elif cfg.posevec.type == 'quaternion':
            #         dst_posevec = dst_posevec.reshape(list(dst_posevec.shape[:-1])+[dst_posevec.shape[-1]//3,3]) #*,N,3
            #         rest_quaternion = axis_angle_to_quaternion(torch.zeros_like(dst_posevec))
            #         pose_quaternion = axis_angle_to_quaternion(dst_posevec)
            #         dst_posevec = pose_quaternion-rest_quaternion
            #         dst_posevec = dst_posevec.reshape(list(dst_posevec.shape[:-2])+[-1])
            # non_rigid_mlp_input.append(dst_posevec)

        # if cfg.non_rigid_motion_mlp.time_input:
        #     if cfg.non_rigid_motion_mlp.time_embed == 'vocab':
        #         time_vec = self.time_embed_fn(frame_id[None,...])
        #     else:
        #         time_vec = self.time_embed_fn(frame_id[None,...]/cfg.non_rigid_motion_mlp.time_vocab_n)[None,...] #N,d
        #     non_rigid_mlp_input.append(time_vec)

        # if non_rigid_mlp_input != []:
        #     non_rigid_mlp_input = torch.cat(non_rigid_mlp_input, dim=-1) #B,D

        #     if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
        #         # mask-out non_rigid_mlp_input 
        #         non_rigid_mlp_input = torch.zeros_like(non_rigid_mlp_input) * non_rigid_mlp_input
        #     non_rigid_mlp_input_dict['condition_code'] = non_rigid_mlp_input

        # if cfg.canonical_mlp.time_input:
        #     if cfg.canonical_mlp.time_embed == 'vocab':
        #         time_vec_cnl = self.time_embed_fn_cnl(frame_id[None,...])
        #     else:
        #         time_vec_cnl = self.time_embed_fn_cnl(frame_id[None,...]/cfg.canonical_mlp.time_vocab_n)[None,...] #N,d
        #     non_rigid_mlp_input_dict['time_vec_cnl'] = time_vec_cnl
        # if pose_condition_cmlp is not None:
        #     non_rigid_mlp_input_dict['condition_code_cmlp'] = pose_condition_cmlp[None,...]

        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "dir_embed_fn": self.dir_embed_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            # "non_rigid_mlp_input": non_rigid_mlp_input_dict,
            # "pose_latent": dst_posevec, 
        })

        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter and \
            kwargs.get('non_rigid_motion_pose_condition',None) is not None:
            kwargs['non_rigid_motion_pose_condition'] = torch.zeros_like(kwargs['non_rigid_motion_pose_condition'])
        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter and \
            kwargs.get('non_rigid_motion_posedelta_condition',None) is not None:
            kwargs['non_rigid_motion_posedelta_condition'] = torch.zeros_like(kwargs['non_rigid_motion_posedelta_condition'])            

        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs, 
                                            dst_Ts=dst_Ts, 
                                            cnl_gtfms=cnl_gtfms)
        motion_weights_vol = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors)
        self.motion_weights_vol=motion_weights_vol[0] # remove batch dimension

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'motion_weights_vol': self.motion_weights_vol
        })

        rays_o, rays_d, rays_d_camera = rays
        rays_shape = rays_d.shape

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        rays_d_camera = torch.reshape(rays_d_camera, [-1,3]).float()

        packed_ray_infos = torch.cat([rays_o, rays_d, rays_d_camera, near, far], -1)
        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
        for k in all_ret:
            if type(all_ret[k])==list:
                all_ret[k] = [torch.reshape(x, list(rays_shape[:-1])+list(x.shape[1:])) for x in all_ret[k]] #merge from all gpus
            else:
                k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])  #3, (num_head?)
                all_ret[k] = torch.reshape(all_ret[k], k_shape)

        if os.environ.get('RETURN_POSE','False').lower()=='true':
            all_ret['POSE'] = pose_refine_output
        return all_ret
