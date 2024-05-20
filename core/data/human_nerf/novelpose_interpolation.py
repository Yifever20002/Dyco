from core.data.human_nerf import train
import os, cv2, pickle, numpy as np
from core.utils.file_util import list_files, split_path
from core.utils.image_util import load_image
from configs import cfg
class Dataset(train.Dataset):
    def __init__(self, subject, **kwargs):
        self.subject = subject
        super().__init__(
            **kwargs)
        #self.image_dir = os.path.join(self.dataset_path, f'images_pose{self.pose_id}')

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):
        new_mesh_path = os.path.join(self.dataset_path, f'mesh_infos_pose{self.pose_id}.pkl')
        print(f'Load novel pose from {new_mesh_path}')
        with open(new_mesh_path, 'rb') as f:
            mesh_infos = pickle.load(f)
        #bbox already in mesh_infos_pose   
        # mesh_infos = 
        # for frame_name in mesh_infos.keys():
        #     bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
        #     mesh_infos[frame_name]['bbox'] = bbox 
        return mesh_infos  

    # def load_train_frames(self):
    #     img_paths = list_files(os.path.join(self.dataset_path, f'images_pose{self.pose_id}'),
    #                            exts=['.png'])
    #     return [split_path(ipath)[1] for ipath in img_paths]

    def load_train_cameras(self):
        cameras = None
        with open(os.path.join(self.dataset_path, f'cameras_pose{self.pose_id}.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras


    def __getitem__(self, idx):
        results = {}

        if cfg.multihead.split == 'view':
            if cfg.test.head_id == -1: #auto
                results['head_id'] = int(cfg.test.head_id) #multiple outputs
                '''
                view_id = self.parse_view_from_frame(frame_name)
                results['head_id'] = self.views.index(view_id)
                '''
            else:
                results['head_id'] = int(cfg.test.head_id)
                raise  
        else:
            results['head_id'] = -1

        bgcolor = np.array(self.bgcolor, dtype='float32')

        H = W = self.img_size

        # load t-pose
        dst_bbox = self.canonical_bbox.copy()
        dst_poses = np.zeros(72, dtype='float32')
        dst_skel_joints = self.canonical_joints.copy()

        # rotate body
        angle = 2 * np.pi / self.total_frames * idx
        add_rmtx = cv2.Rodrigues(np.array([0, -angle, 0], dtype='float32'))[0]
        root_rmtx = cv2.Rodrigues(dst_poses[:3])[0]
        new_root_rmtx = add_rmtx@root_rmtx
        dst_poses[:3] = cv2.Rodrigues(new_root_rmtx)[0][:, 0]

        # rotate boundinig box
        dst_bbox = self.rotate_bbox(dst_bbox, add_rmtx)

        K = self.camera['K'].copy()
        E = self.camera['E'].copy()
        R = E[:3, :3]
        T = E[:3, 3]

        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T) 
        rays_o = rays_o.reshape(-1, 3)# (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')
    
        batch_rays = np.stack([rays_o, rays_d, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_skel_joints
                )
            cnl_gtfms = get_canonical_global_tfms(self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })                                    

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = self.motion_weights_priors.copy()

        if 'cnl_bbox' in self.keyfilter:
            min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })
        
        return results
        
    # def load_image(self, frame_name, bg_color):
    #     imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
    #     orig_img = np.array(load_image(imagepath))

    #     maskpath = os.path.join(self.dataset_path, 
    #                             'masks', 
    #                             '{}.png'.format(frame_name))
    #     alpha_mask = np.array(load_image(maskpath))
        
    #     # undistort image
    #     if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
    #         K = self.cameras[frame_name]['intrinsics']
    #         D = self.cameras[frame_name]['distortions']
    #         orig_img = cv2.undistort(orig_img, K, D)
    #         alpha_mask = cv2.undistort(alpha_mask, K, D)

    #     alpha_mask = alpha_mask / 255.
    #     #img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
    #     img = orig_img
    #     if cfg.resize_img_scale != 1.:
    #         img = cv2.resize(img, None, 
    #                             fx=cfg.resize_img_scale,
    #                             fy=cfg.resize_img_scale,
    #                             interpolation=cv2.INTER_LANCZOS4)
    #         alpha_mask = cv2.resize(alpha_mask, None, 
    #                                 fx=cfg.resize_img_scale,
    #                                 fy=cfg.resize_img_scale,
    #                                 interpolation=cv2.INTER_LINEAR)
                                
    #     return img, alpha_mask #TODO, alpha_mask should be useless here
