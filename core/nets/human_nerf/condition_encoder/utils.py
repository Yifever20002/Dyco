from configs import cfg 
import torch, numpy as np
from core.nets.human_nerf.condition_encoder.seq_encoder import PoseSeq_Encoder, RGBSeq_Encoder

    

def create_condition_encoder(network, **kwargs):
    if network=='PoseSeq_Encoder':
        network_cfg = kwargs.pop(network)
        encoder = PoseSeq_Encoder(**kwargs,**network_cfg)
    elif network=='RGBSeq_Encoder':
        network_cfg = kwargs.pop(network)
        encoder = RGBSeq_Encoder(
            length=cfg.rgb_history.length,
            **kwargs,**network_cfg)
    else:
        raise ValueError
    return encoder