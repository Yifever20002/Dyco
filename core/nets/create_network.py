import os, torch
import imp

from configs import cfg
from utils import custom_print

def _query_network():
    module = cfg.network_module   #'core.nets.human_nerf.network'
    module_path = module.replace(".", "/") + ".py"
    network = imp.load_source(module, module_path).Network    #Network
    return network


def create_network():
    network = _query_network()
    network = network()     #Network()
    if cfg.modules.pretrained_path != 'empty':     #load pretrained model
        assert os.path.isfile(cfg.modules.pretrained_path), cfg.modules.pretrained_path
        state_dict = torch.load(cfg.modules.pretrained_path, map_location='cpu')['network']
        
        if int(os.environ.get('LOAD_D',0))==1:
            state_dict['cnl_mlp.module.output_linear_density.0.weight'] = state_dict['cnl_mlp.module.output_linear.0.weight'][-1:,:]
            state_dict['cnl_mlp.module.output_linear_density.0.bias'] = state_dict['cnl_mlp.module.output_linear.0.bias'][-1:]
        
        if int(os.environ.get('LOAD_C',0))==1:
            assert 'cnl_mlp.module.output_linear_rgb.0.weight' not in state_dict \
            and 'cnl_mlp.module.output_linear_rgb.0.weight' in network.state_dict()
            state_dict['cnl_mlp.module.output_linear_rgb.0.weight'] = state_dict['cnl_mlp.module.output_linear.0.weight'][:-1,:]
            state_dict['cnl_mlp.module.output_linear_rgb.0.bias'] = state_dict['cnl_mlp.module.output_linear.0.bias'][:-1]

        if cfg.modules.canonical_mlp.reinit:
            custom_print('Reinitialize canonical_mlp')
            state_dict = {k:v for k,v in state_dict.items() if not 'cnl_mlp' in k}
        if cfg.modules.non_rigid_motion_mlp.reinit:
            custom_print('Reinitialize rigid_motion_mlp')
            state_dict = {k:v for k,v in state_dict.items() if not 'non_rigid_mlp' in k}
        msg = network.load_state_dict(state_dict, strict=False)
        custom_print(msg)
        for name, param in network.named_parameters():
            param.requires_grad = False
            if cfg.modules.canonical_mlp.tune and 'cnl_mlp' in name:
                param.requires_grad = True
            if cfg.modules.non_rigid_motion_mlp.tune and 'non_rigid_mlp' in name:
                param.requires_grad = True
            if cfg.modules.canonical_mlp.tune and 'cnl_mlp' in name:
                param.requires_grad = True

            if cfg.modules.canonical_mlp.tune_last>=0 and 'cnl_mlp' in name:
                if 'cnl_mlp.module.output_linear' in name:
                    param.requires_grad = True
                else:
                    i = int(name.split('cnl_mlp.module.pts_linears.')[1].split('.')[0])
                    if (14-i)//2<int(cfg.modules.canonical_mlp.tune_last):
                        param.requires_grad = True
            if int(os.environ.get('TUNE_C',0))==1 and 'cnl_mlp.module.output_linear_rgb' in name:  
                param.requires_grad = True
            if int(os.environ.get('TUNE_D',0))==1 and 'cnl_mlp.module.output_linear_density' in name:  
                param.requires_grad = True
            if cfg.modules.pose_decoder.tune and 'pose_decoder' in name:
                param.requires_grad = True 
            if cfg.modules.mweight_vol_decoder.tune and 'mweight_vol_decoder' in name:
                param.requires_grad = True 
    return network
