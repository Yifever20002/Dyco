import torch
import torch.nn as nn

from core.utils.network_util import initseq
from core.nets.human_nerf.condition_encoder.utils import create_condition_encoder
from configs import cfg
class NonRigidMotionMLP(nn.Module):
    def __init__(self,
                 pos_embed_size=3, 
                 mlp_width=128,
                 mlp_depth=6,
                 mlp_depth_plus=0,
                 skips=None, 
                 last_linear_scale=1,**kwargs):
        super(NonRigidMotionMLP, self).__init__()

        self.skips = [4] if skips is None else skips
        self.skips_all = cfg.non_rigid_motion_mlp.skips_all
        if cfg.non_rigid_motion_mlp.xyz_encoder.depth>0:
            depth, width = cfg.non_rigid_motion_mlp.xyz_encoder.depth, cfg.non_rigid_motion_mlp.xyz_encoder.width
            self.xyz_encoder = [nn.Linear(pos_embed_size,width), nn.ReLU()]
            for _ in range(depth-1):
                self.xyz_encoder.extend([nn.Linear(width,width), nn.ReLU()])
            self.xyz_encoder = nn.Sequential(*self.xyz_encoder)
            pos_embed_size = width
        else:
            self.xyz_encoder = None 

        if cfg.non_rigid_motion_mlp.pose_condition.length>0:
            self.pose_condition_encoder = create_condition_encoder(**cfg.non_rigid_motion_mlp.pose_condition)
            pose_condition_dim = self.pose_condition_encoder.output_dim
        else:
            self.pose_condition_encoder = None
            pose_condition_dim = 0

        if cfg.non_rigid_motion_mlp.posedelta_condition.length>0:
            self.posedelta_condition_encoder = create_condition_encoder(**cfg.non_rigid_motion_mlp.posedelta_condition)
            posedelta_condition_dim = self.posedelta_condition_encoder.output_dim

        else:
            self.posedelta_condition_encoder = None
            posedelta_condition_dim = 0

        block_mlps = [nn.Linear(pos_embed_size+pose_condition_dim+posedelta_condition_dim, 
                                mlp_width), nn.ReLU()]
        
        layers_to_cat_inputs = []
        for i in range(1, mlp_depth+mlp_depth_plus):
            if i in self.skips:
                layers_to_cat_inputs.append(len(block_mlps))
                if self.skips_all:
                    block_mlps += [nn.Linear(mlp_width+pos_embed_size+pose_condition_dim+posedelta_condition_dim, mlp_width), 
                                nn.ReLU()]
                else:
                    block_mlps += [nn.Linear(mlp_width+pos_embed_size, mlp_width), 
                                nn.ReLU()]
            else:
                if i>=mlp_depth-1:
                    if i==mlp_depth-1:
                        block_mlps += [nn.Linear(mlp_width, mlp_width*last_linear_scale), nn.ReLU()]
                    else:
                        block_mlps += [nn.Linear(mlp_width*last_linear_scale, mlp_width*last_linear_scale), nn.ReLU()]
                else:
                    block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        block_mlps += [nn.Linear(mlp_width*last_linear_scale, 3)] 

        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)

        self.layers_to_cat_inputs = layers_to_cat_inputs

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros 
        init_val = 1e-5
        last_layer = self.block_mlps[-1] 
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()             


    def forward(self, pos_embed, pos_xyz, pose_condition, posedelta_condition=None, viewdirs=None, weights=None, **_):
        # condition_code = condition_code.expand((pos_embed.shape[0],-1)) #P,D
        # condition_code = localize_condition_code(condition_code, weights)
        # h = torch.cat([condition_code, pos_embed], dim=-1)
        if self.xyz_encoder is not None:
            pos_embed = self.xyz_encoder(pos_embed) #(N,D)

        h = [pos_embed]
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
        h = torch.cat(h, dim=-1)
        input_embed = h

        if viewdirs is not None:
            h = torch.cat([h, viewdirs], dim=-1)
        for i in range(len(self.block_mlps)):
            if i in self.layers_to_cat_inputs:
                if self.skips_all:
                    h = torch.cat([h, input_embed], dim=-1)
                else:
                    h = torch.cat([h, pos_embed], dim=-1)
            h = self.block_mlps[i](h)
        trans = h
        result = {
            'xyz': pos_xyz + trans,
            'offsets': trans
        }                   
        return result
