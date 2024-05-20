import torch, os
import torch.nn as nn

from core.utils.network_util import initseq
from core.nets.human_nerf.condition_encoder.utils import create_condition_encoder
from configs import cfg 


class CanonicalMLP(nn.Module):
    def __init__(self, mlp_depth=8, mlp_width=256, 
                 pos_embed_dim=3, skips=None, 
                 view_dir=False, input_ch_dir=3, 
                 pose_ch=69,
                 pose_color='wo',
                 mlp_depth_plus=0,
                 last_linear_scale=1,  **kwargs):
        super(CanonicalMLP, self).__init__()

        if skips is None:
            skips = [2]
        self.skips_all = cfg.canonical_mlp.skips_all
        self.mlp_width = mlp_width
        self.view_dir = view_dir
        self.pose_color = pose_color #'dependent','ao' or 'wo'
        self.pose_ch = pose_ch
        self.input_ch_dir = input_ch_dir

        if cfg.canonical_mlp.xyz_encoder.depth>0:
            depth, width = cfg.canonical_mlp.xyz_encoder.depth, cfg.canonical_mlp.xyz_encoder.width
            self.xyz_encoder = [nn.Linear(pos_embed_dim,width), nn.ReLU()]
            for _ in range(depth-1):
                self.xyz_encoder.extend([nn.Linear(width,width), nn.ReLU()])
            self.xyz_encoder = nn.Sequential(*self.xyz_encoder)
            pos_embed_dim = width
        else:
            self.xyz_encoder = None 
        
        if cfg.canonical_mlp.triplane:# 32-d features for 4 resolutions
            input_dim = 128+pos_embed_dim
        else:
            input_dim = pos_embed_dim
        if cfg.canonical_mlp.pose_condition.length>0:
            self.pose_condition_encoder = create_condition_encoder(**cfg.canonical_mlp.pose_condition)
            input_dim += self.pose_condition_encoder.output_dim
        else:
            self.pose_condition_encoder = None
        if cfg.canonical_mlp.posedelta_condition.length>0:
            self.posedelta_condition_encoder = create_condition_encoder(**cfg.canonical_mlp.posedelta_condition)
            input_dim += self.posedelta_condition_encoder.output_dim
        else:
            self.posedelta_condition_encoder = None
        
        if cfg.rgb_history.length>0:
            self.rgb_condition_encoder = create_condition_encoder(**cfg.canonical_mlp.rgb_condition)
            if cfg.canonical_mlp.rgb_condition.input_layer == 0:
                input_dim += self.rgb_condition_encoder.output_dim
        else:
            self.rgb_condition_encoder = None

        pts_block_mlps = [nn.Linear(input_dim, mlp_width), nn.ReLU()]
        layers_to_cat_input = []
        for i in range(mlp_depth+mlp_depth_plus-1):
            if i in skips:
                layers_to_cat_input.append(len(pts_block_mlps))
                if self.skips_all:
                    pts_block_mlps += [nn.Linear(mlp_width + input_dim, mlp_width), 
                                    nn.ReLU()]                    
                else:
                    pts_block_mlps += [nn.Linear(mlp_width + pos_embed_dim, mlp_width), 
                                    nn.ReLU()]
            else:
                if i>=mlp_depth-2:
                    if i==mlp_depth-2:
                        pts_block_mlps += [nn.Linear(mlp_width, mlp_width*last_linear_scale), nn.ReLU()]
                    else:
                        pts_block_mlps += [nn.Linear(mlp_width*last_linear_scale, mlp_width*last_linear_scale), nn.ReLU()]
                else:
                    pts_block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        self.layers_to_cat_input = layers_to_cat_input

        self.pts_linears = nn.ModuleList(pts_block_mlps)
        initseq(self.pts_linears)

        # output: rgb + sigma (density)
        if self.view_dir or self.pose_color == 'direct':
            self.output_linear_density = nn.Sequential(nn.Linear(mlp_width, 1))
            self.output_linear_rgb_1 = nn.Sequential(nn.Linear(mlp_width, mlp_width))
            dim = mlp_width
            if self.view_dir:
                dim += self.input_ch_dir
            if self.pose_color=='direct':
                dim += self.pose_ch
            self.output_linear_rgb_2 = nn.Sequential(
                nn.Linear(dim, mlp_width),
                nn.Linear(mlp_width, 3))
        else:
            self.output_linear = nn.Sequential(nn.Linear(mlp_width*last_linear_scale, 4))
            initseq(self.output_linear)

        if self.pose_color == 'ao':
            self.output_linear_ao_1 = nn.Sequential(nn.Linear(mlp_width, mlp_width))
            dim = mlp_width+self.pose_ch
            self.output_linear_ao_2 = nn.Sequential(
                nn.Linear(dim, mlp_width),
                nn.Linear(mlp_width, 1)) #output a scalar
            self.ao_activation = torch.nn.Sigmoid()

    def forward(self, features_xyz, pos_xyz, pos_embed, dir_embed=None, pose_condition=None, 
                posedelta_condition=None, 
                pose_latent=None, weights=None, iter_val=1e7, 
                rgb_condition=None, visible=None, **_):
        if cfg.canonical_mlp.triplane:
            h = [features_xyz]
        else:
            h = []
        if self.xyz_encoder is not None:
            pos_embed = self.xyz_encoder(pos_embed) #(N,D)
        h.append(pos_embed)
        if self.pose_condition_encoder is not None:
            pose_condition_embed = self.pose_condition_encoder(
                x=pose_condition.expand((pos_embed.shape[0],)+pose_condition.shape[1:]),  #N_point, T, 23, D0
                weights=weights.detach()) #N_point, 24
            h.append(pose_condition_embed)
        if self.posedelta_condition_encoder is not None:
            posedelta_condition_embed = self.posedelta_condition_encoder(
                x=posedelta_condition.expand((pos_embed.shape[0],)+posedelta_condition.shape[1:]),  #N_point, T, 23, D0
                weights=weights.detach()) #N_point, 24
            h.append(posedelta_condition_embed)        
        if self.rgb_condition_encoder is not None:
            rgb_condition_embed = self.rgb_condition_encoder(x=rgb_condition, visible=visible) #N,T,V,D
            if cfg.canonical_mlp.rgb_condition.input_layer == 0:
                h.append(rgb_condition_embed)

        h = torch.cat(h, dim=-1)
        input_embed = h
        for i, _ in enumerate(self.pts_linears):
            if i in self.layers_to_cat_input:
                if self.skips_all:
                    h = torch.cat([input_embed, h], dim=-1)
                else:
                    h = torch.cat([pos_embed, h], dim=-1)
            h = self.pts_linears[i](h)

        if self.view_dir or self.pose_color=='direct':
            density = self.output_linear_density(h)
            features = [self.output_linear_rgb_1(h)] #N,D
            if self.view_dir:
                features.append(dir_embed)
            if self.pose_color=='direct':
                features.append(pose_latent)

            rgb = self.output_linear_rgb_2(torch.cat(features,dim=1))
            outputs = torch.cat([rgb, density],dim=1) #N, 4
        else:
            outputs = self.output_linear(h)
        
        if self.pose_color == 'ao':
            feature = self.output_linear_ao_1(h)
            ao = self.output_linear_ao_2(torch.cat([feature, pose_latent], axis=1))
            ao = self.ao_activation(ao)
            outputs = torch.cat([rgb*ao, density],dim=1) #N, 4

        return outputs    
        