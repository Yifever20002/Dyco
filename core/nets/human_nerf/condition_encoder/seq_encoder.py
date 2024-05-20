import torch, os
import torch.nn as nn
from core.utils.network_util import initseq
from core.nets.human_nerf.rgb_feature import featurename2dim
from configs import cfg 
import numpy as np

if os.path.isfile(cfg.localize_part2joints_file):
    PART2JOINTS = np.load(cfg.localize_part2joints_file)
    PART2JOINTS = torch.tensor(PART2JOINTS).cuda() #24,23
N_JOINT = 23


representation2dim = {
    'axis-angle': 3,
    'quaternion': 4,
    'matrix': 9,
}

def localize_seq_code(x, weights, enable, fg_threshold):
    '''
    Input:
        x: (N,T,23,D0)
        weights: (N,24)
    Output:
        y: (N,T,23&M,D0)
        is_fg (N,1)
    '''
    weights = weights.detach()
    if enable==False:
        return x, torch.ones_like(weights[:,0:1]) #(N,1)
    part = torch.argmax(weights,dim=1) #N
    joints_mask = PART2JOINTS[part] #24,23 -> N,23
    is_fg = weights.max(axis=1,keepdims=True)[0]>fg_threshold  #(N,1)
    
    joints_mask = joints_mask*is_fg #(N,23)
    if x.shape[2] == 23:
        y = x*joints_mask[:,None,:,None] #(N,T,23,D0) * (N,1,23,1) -> (N,T,23,D0)
    else:
        local_pose, RhTh = torch.split(x, [23, 2], dim=2)
        y = local_pose*joints_mask[:,None,:,None]
        y = torch.cat([y, RhTh], dim=2)
    return y, is_fg


class PoseSeq_Encoder(nn.Module):
    def __init__(self, length, representation, localize, 
                    D1, D2, 
                    bg_condition, name, **kwargs):
        super(PoseSeq_Encoder, self).__init__()
        self.localize, self.bg_condition = localize, bg_condition
        assert representation in representation2dim
        assert self.bg_condition in ['zero_output', 'zero_input']
        self.input_dim = representation2dim[representation]*N_JOINT
        if name == 'posedelta_condition':
            self.input_dim += representation2dim[representation]*2
        if D1>0:
            self.mlp1 = nn.Sequential(nn.Linear(self.input_dim,D1), nn.ReLU())
        else:
            self.mlp1 = nn.Identity()
            D1 = self.input_dim
        if D2>0:
            self.mlp2 = nn.Sequential(nn.Linear(D1*length, D2), nn.ReLU())
        else:
            self.mlp2 = nn.Identity()
            D2 = D1*length
        self.output_dim = D2

    def forward(self, x, weights):
        '''
        x: (N,T,23,D0)
        weights: (N,24)

        return output (N,self.output_dim)
        '''
        
        N, T, _, D0 = x.shape
        x, is_fg = localize_seq_code(x, weights, **self.localize) #(N,T,23,D0) (N,1)
        #Note that if localize['enable']==False, then is_fg=all_ones
        x_joint_flat = x.view(N,T,-1)
        x1 = self.mlp1(x_joint_flat) # (N,T,23*D0)->(N,T,D1)

        x1_seq_flat = x1.view(N,-1) #(N,T*D1)
        x2 = self.mlp2(x1_seq_flat) #(N,D2)

        if self.bg_condition == 'zero_output' and self.localize['enable']==True: 
            x2 = x2*is_fg #(N,self.output_dim)

        return x2

class RGBSeq_Encoder(nn.Module):
    def __init__(self, length, D1, D2, view_reduce,
            #fg_threshold, bg_condition,
            **kwargs):
        super(RGBSeq_Encoder, self).__init__()
        if cfg.rgb_history.feature_name in featurename2dim:
            self.input_dim = featurename2dim[cfg.rgb_history.feature_name]
        elif cfg.rgb_history.feature_name == 'resnet-scratch':
            self.input_dim = cfg.rgb_history.feature_net.out_chs[-1]
        else:
            raise ValueError

        self.view_reduce = view_reduce
        # self.fg_threshold, self.bg_condition = fg_threshold, bg_condition
        if D1>0:
            self.mlp1 = nn.Sequential(nn.Linear(self.input_dim,D1), nn.ReLU())
        else:
            self.mlp1 = nn.Identity()
            D1 = self.input_dim
        if D2>0:
            self.mlp2 = nn.Sequential(nn.Linear(D1*length, D2), nn.ReLU())
        else:
            self.mlp2 = nn.Identity()
            D2 = D1*length
        self.output_dim = D2

    def forward(self, x, visible=None):
        '''
        x: (N,T,V,D0)
        weights: (N,24) used to determine fg or bg point
        return output (N,self.output_dim)
        '''
        
        N, T, V, D0 = x.shape #TODO how to handle view
        x1 = self.mlp1(x) # (N,T,V,D0)->(N,T,V,D1)

        if self.view_reduce == 'mean_after_mlp1':
            if cfg.rgb_history.view_selection == 'visible':
                #visible = (torch.rand_like(visible.float())>0.8)# DEBUG
                #x1 = torch.sum(x1*visible, dim=2)/(torch.sum(visible, dim=2)+1e-10) #dim= #(N,T,D) This may lead to numerical in-stability
                #x1 = torch.zeros_like(x1)#DEBUG!
                x1 = torch.sum(x1*visible, dim=2) #DEBUG
            else:
                x1 = torch.mean(x1, dim=2) #(N,T,D1)
        else:
            raise ValueError

        x1_seq_flat = x1.view(N,-1) #(N,T*D1)
        x2 = self.mlp2(x1_seq_flat) #(N,D2)

        return x2


