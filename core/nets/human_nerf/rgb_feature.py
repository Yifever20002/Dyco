import argparse
import torch
from torchvision import models, transforms
import torch.nn as nn
from configs import cfg
from core.nets.human_nerf.feature_net.resnet import ResNet
from utils import custom_print
transform = transforms.Compose([
    # transforms.Resize((512, 512)),  # Resize images to an appropriate size
    # transforms.ToTensor(),  # Convert images to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

featurename2dim = {
    'rgb': 3,
    'resnet34-layer2': 64,
    'resnet34-layer4': 64,
    'resnet34-layer5': 128,
    'resnet34-layer6': 256,
    'resnet34-layer7': 512,
}

class RGB_FeatureIndexer(nn.Module):
    def __init__(self, 
            feature_name, precompute='empty', h0=512, w0=512, 
            window_size=16,
            from_scratch=False,  **kwargs):
        super(RGB_FeatureIndexer, self).__init__()    
        self.precompute = precompute
        self.window_size = window_size
        if precompute!='empty':
            self.model = None
            self.output_dim = featurename2dim[feature_name]
        else:
            if feature_name == 'resnet-scratch':
                self.model = ResNet(**cfg.rgb_history.feature_net)
                self.model_frozen = False
                self.output_dim = self.model.output_dim
            else:
                net, layer = feature_name.split('-') # e.g. resnet34, layer6
                layer = int(layer.replace('layer',''))
                model = getattr(models, net)(pretrained=True)
                model.eval()
                self.model_frozen = True
                self.model = torch.nn.Sequential(*(list(model.children())[:layer+1]))
                custom_print(f"Load model [{net}], extract features from layer-[{layer}]")
                for param in self.model.parameters():
                    param.requires_grad = False
                self.output_dim = featurename2dim[feature_name]
        self.h0, self.w0 = h0, w0
        assert self.h0==self.w0, 'Only support square images'
    
    def forward_rigid_3Dwarp(self, pts, 
            forward_motion_weights,
            forward_motion_scale_Rs, forward_motion_Ts,):
        total_bases = forward_motion_weights.shape[-1]
        forward_motion_weights_sum = torch.sum(forward_motion_weights, 
                                                dim=-1, keepdim=True) #(N,1)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.einsum('nij,bj->bni', forward_motion_scale_Rs[:,i], pts)+forward_motion_Ts[:,i]  #T,3,3, (N,3)-> (N,T,i)
            weighted_pos = forward_motion_weights[:,None,i:i+1] * pos #(N,1,1) (N,T,3)->(N,T,3)
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / forward_motion_weights_sum.clamp(min=0.0001)[:,None,:] #(b,N,3)/(B,1,1)
        return x_skel

    def project_world2image(self, xyzs, w2cs):
        r""" 

        Args:
            - xyzs: Array (N, T, 3)
            - w2cs: Array (T, V_num, 4, 3)
            
        Returns:
            - Array (N, T, V_num, 2) int32
        """
        N, T = xyzs.shape[:2]
        xyzs = torch.cat([xyzs, torch.ones_like(xyzs[...,-1:])], axis=-1) #(N, T, 4)
        #((1)x,?y, V_num-z, 4-i, 3-j) @ (N_ray-128-x, ?-y, (1)-z, 4) -> (N_ray-128-x, ?-y, V_num-z, 3)
        uvzs = torch.einsum('xyzij,xyzj->xyzi', w2cs[None,...], xyzs[:,:,None,:])  #(N, T, V_num,4)
        depth = uvzs[...,-1:] #
        uvs = uvzs[...,:2]/(uvzs[...,-1:]+1e-10) #(N,T, V_num,2)
        uvs = uvs[...,[1,0]]
        #uvs = torch.clip(uvs, min=0, max=self.h0-1) we leave it for 'visible'
        return uvs, depth

    def forward(self, cnl_pts, 
            forward_motion_weights, 
            forward_motion_scale_Rs_history, forward_motion_Ts_history, 
            w2c_history, #Must be forward!!!
            rgb_history, depth_history=None):
        '''
        Input:
            cnl_pts: (N,3)
            forward_motion_weights: (N, 24)
            forward_motion_scale_Rs: (1,T,24,3,3), forward_motion_Ts (1,T,24)
            w2cs: (1,T,V,4,3)
            rgb_history: (1,T,V,H,W,C)
            depth_history: (1,T,V,H,W)
        Output:
            features: (N,T,V,Do)
        '''
        with torch.no_grad():
            pose_pts = self.forward_rigid_3Dwarp(cnl_pts, 
                    forward_motion_weights,
                    forward_motion_scale_Rs_history[0], forward_motion_Ts_history[0]) #(N,T,3)

            uvs, pts_depth = self.project_world2image(pose_pts, w2c_history[0]) #(N,T,V,2)
            uvs = uvs.int() # TODO bilinear sampling
            N,T,V = uvs.shape[:-1]
            # Something new by Yutong, Nov 26th
            # Using the depth map, 
            # we know whether uvs are visible, on the surface and not occluded.
            if cfg.rgb_history.view_selection=='visible':
                visible = (uvs[...,1]>=0)*(uvs[...,0]>=0)*(uvs[...,0]<self.h0)*(uvs[...,1]<self.w0)
                uvs = torch.clip(uvs, min=0, max=self.h0-1)
                visible = visible[...,None]
                assert depth_history != None
                depth_history_1d = depth_history.view(depth_history.shape[:-2]+(-1,)) #(N,T,V,H*W)
                uvs_1d = uvs[...,0]*self.w0 + uvs[...,1] #(N,T,V)
                surface_depth = torch.gather(
                    input=depth_history_1d[None,...].expand(N,-1,-1,-1), #(N,T,V,H*W)
                    dim=-1,
                    index=uvs_1d[...,None].long(), #(N,T,V,1)
                ) #(N,T,V,1)
                # surface_depth (N,T,V,1), pts_depth (N,T,V,1)
                visible *= torch.abs(pts_depth-surface_depth)<cfg.rgb_history.visible_threshold #(N,T,V,1)
                '''
                #DEBUG
                import ipdb; ipdb.set_trace()
                import numpy as np
                import cv2
                vis = torch.sum(visible, axis=2) #(N,T,1)
                for ii,dd in enumerate(depth_history[0]):
                    dm = dd>0.5
                    min_ = dd[dd>0.5].min()
                    dd = torch.clip((dd.max()-dd)/(dd.max()-min_),0,1)
                    dd = ((dd*dm).cpu().numpy()*255).astype(np.uint8)
                    cv2.imwrite(f'debug_output/depth_{ii}.png', dd)
                '''
            else:
                uvs = torch.clip(uvs, min=0, max=self.h0-1)
                visible = None

        imgs = rgb_history.reshape((-1,)+rgb_history.shape[3:]) #T,V,H,W,C
        uvs = uvs.reshape(N, T*V, 2) #N,T*V,2

        imgs = imgs.permute((0,3,1,2)) #(T*V,C,H,W)
        if self.precompute != 'empty':
            x = imgs
        else:
            x = transform(imgs)
            if self.model_frozen:
                with torch.no_grad():
                    x = self.model(x)
            else:
                x = self.model(x)
        
        hi, wi = x.shape[-2], x.shape[-1] 
        scale_h, scale_w = int(self.h0/hi), int(self.w0/wi) #2,4,8,16

        assert self.window_size>=scale_h #pooling

        if self.window_size>scale_h:
            assert self.window_size%scale_h==0
            s = int(self.window_size/scale_h)
            if cfg.rgb_history.query_type == 'v1':
                x = torch.nn.functional.avg_pool2d(x, kernel_size=s, stride=s, padding=0, count_include_pad=False) #(T*V, C, H,W)
                hi, wi = x.shape[-2], x.shape[-1] 
                scale_h, scale_w = int(self.h0/hi), int(self.w0/wi) #2,4,8,16
                assert scale_h==self.window_size
                indices_i = (torch.floor(uvs[...,0]/scale_h)*wi+torch.floor(uvs[...,1]/scale_w)).long() #N,T*V
            elif cfg.rgb_history.query_type == 'v2':
                x = torch.nn.functional.avg_pool2d(x, kernel_size=s+1, stride=1, padding=s//2, count_include_pad=False) #make it an odd number
                indices_i = (torch.floor(uvs[...,0]/scale_h)*wi+torch.floor(uvs[...,1]/scale_w)).long() #N,T*V

        x_flatten = x.reshape(x.shape[:2]+(-1,)) #T*V,D,h*w
        x_indexed = torch.gather(
            input = x_flatten[None,:,:,:].expand(N,-1,-1,-1), #N,T*V,D,h*w,
            dim = -1,
            index = indices_i[:,:,None,None].expand(-1,-1,self.output_dim,-1) #N,T*V,D,1
        ) #->(N,TV,D,1)
    
        x_indexed = x_indexed.view(N, T, V, self.output_dim)

        #Sanity Check
        '''
        if forward_motion_weights.max()>0.8:
            import ipdb; ipdb.set_trace()
            #pi = forward_motion_weights.argmax().item()//24
            pi = visible[:,0,0,0].int().argmax()
            vis = visible[pi,0,:,0]
            uv_ = uvs[pi] #view_num, 2
            #vis = visible[pi,0,:,0] #V
            import cv2
            #for view, uv in zip([1,7,13,19],uv_):
            for view, uv, v in zip([0,2,4,6,8],uv_, vis):
                fn = 270
                #img = cv2.resize(cv2.imread(f'data/zju/CoreView_387/Camera_B{view}/{fn:06}.jpg'),(512,512))
                img = cv2.imread(f'dataset/pjlab_mocap/xianbei_v1.0-train/images/frame_{fn:06}_view_{view:02}.png')
                img = cv2.circle(img,(uv[1].item(),uv[0].item()),radius=8, color=(0,255,0))
                cv2.imwrite(f'debug_output/view{view}_{fn:06}_visible{v.item()}.jpg', img)
            import ipdb; ipdb.set_trace()
        '''
        return x_indexed, visible 

        



        

        
                
        

