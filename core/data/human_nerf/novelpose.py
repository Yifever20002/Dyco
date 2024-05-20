from core.data.human_nerf import train
import os, cv2, pickle, numpy as np
from core.utils.file_util import list_files, split_path
from core.utils.image_util import load_image
from configs import cfg
class Dataset(train.Dataset):
    def __init__(self, subject, pose_id, **kwargs):
        self.pose_id = pose_id
        self.subject = subject
        super().__init__(
            **kwargs)
        self.image_dir = os.path.join(self.dataset_path, f'images_pose{self.pose_id}')

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
 
        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox 
        return mesh_infos  

    def load_train_frames(self):
        img_paths = list_files(os.path.join(self.dataset_path, f'images_pose{self.pose_id}'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]

    def load_train_cameras(self):
        cameras = None
        with open(os.path.join(self.dataset_path, f'cameras_pose{self.pose_id}.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras



    def load_image(self, frame_name, bg_color):
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))
        
        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        #img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        img = orig_img
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask #TODO, alpha_mask should be useless here
