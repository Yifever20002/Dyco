import sys
from attrdict import AttrDict
import cv2
import torch
sys.path = ['RAFT/core']+sys.path
from RAFT.core.raft import RAFT
from RAFT.core.raft_utils.utils import InputPadder
import numpy as np

def get_view_frame(basename):
    basename = basename.replace('.jpg','').replace('.png','')
    if '(' in basename:
        #'Camera (5)-CoreView_313_Camera_(5)_0001_2019-08-23_16-08-50.592.jpg.png'
        view = int(basename.split(')-')[0][8:])
        frame = int(basename.split('_')[-3])
        frame = frame -1 
        return view, frame
    elif '-' in basename: #ZJU # 'Camera_B16-000180'
        view = int(basename.split('-')[0][len('Camera_B'):])
        frame = int(basename.split('-')[1])
        return view, frame
    elif 'frame_' in basename: #I3D 'frame_000951_view_02'
        frame, view = int(basename.split('_')[1]), int(basename.split('_')[-1])
        return view, frame
class DME_Computer(object):

    def __init__(self):
        raft_model_name='things' #sys.argv[1]
        raftargs = AttrDict({'model':f'RAFT/models/raft-{raft_model_name}.pth', 
                        'small':False, 
                        'mixed_precision':False})
        self.raft_iterations=20

        self.raft_model = RAFT(raftargs)
        dict_ = torch.load(raftargs.model, map_location=torch.device('cpu'))
        dict_ = {k.replace('module.',''):v for k,v in dict_.items()}
        self.raft_model.load_state_dict(dict_)
        self.raft_model = self.raft_model.cuda()
        self.raft_model.eval()

        self.last_item = None
        self.metrics = []
        self.of_threshold = 1

    def compute_optical_flow(self, img1, img2):
        #0-255?
        with torch.no_grad():
            img1 = img1.permute(2, 0, 1).float()[None].to('cuda')
            img2 = img2.permute(2, 0, 1).float()[None].to('cuda')
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)
            flow_low, flow_up = self.raft_model(img1, img2, iters=self.raft_iterations, test_mode=True)
        return flow_up[0]

    def compute_epe(self, f1, f2, mask=None):
        epe = torch.norm(f1 - f2, dim=0)
        if mask is not None:
            return epe[mask].mean()
        else:
            return epe.mean()
    
    def compute_mask_from_rgb(self, img):
        #img (H,W,3) (assume black background)
        mask = torch.sum(img, axis=-1)>0
        return mask

    def append(self, gt, pred, name):
        # The input has to be ordered
        gt, pred = (gt*255).int(), (pred*255).int()
        view, frame = get_view_frame(name)
        item = {'gt':gt, 'pred':pred, 'view': view, 'frame':frame}
        if self.last_item is None:
            self.last_item = item
            return None
        else:
            if self.last_item['view'] == item['view'] and \
                    (item['frame']-self.last_item['frame'])<=60:
                gt_of = self.compute_optical_flow(self.last_item['gt'], item['gt'])
                gt_mask = self.compute_mask_from_rgb(item['gt'])
                gt_of[:,gt_mask==False] = 0 #Set bg to 0

                pred_of = self.compute_optical_flow(self.last_item['pred'], item['pred'])
                pred_mask = self.compute_mask_from_rgb(item['pred'])
                pred_of[:,pred_mask==False] = 0

                gt_of_motion_mask = torch.sum(torch.abs(gt_of), axis=0)>self.of_threshold #H,W
                if gt_of_motion_mask.sum()==0:
                    self.last_item = item
                    return None #Remove this frame from evaluation
                pred_of_motion_mask = torch.sum(torch.abs(pred_of), axis=0)>self.of_threshold
                m_ = (gt_of_motion_mask + pred_of_motion_mask)>0
                epe = self.compute_epe(gt_of, pred_of, mask=gt_of_motion_mask).item()
                self.metrics.append(epe)
                self.last_item = item
                return epe
            else:
                self.last_item = item
                return None

    def finalize(self):
        return np.mean(self.metrics)
