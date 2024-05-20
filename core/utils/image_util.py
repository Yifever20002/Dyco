import os
import shutil

from termcolor import colored
from PIL import Image
import numpy as np
import imageio, pickle
import torch


def load_image(path, to_rgb=True):
    img = Image.open(path)
    return img.convert('RGB') if to_rgb else img


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def to_8b_image(image):
    return (255.* np.clip(image, 0., 1.)).astype(np.uint8)


def to_3ch_image(image):
    if len(image.shape) == 2:
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        assert image.shape[2] == 1
        return np.concatenate([image, image, image], axis=-1)
    else:
        print(f"to_3ch_image: Unsupported Shapes: {len(image.shape)}")
        return image


def to_8b3ch_image(image):
    return to_3ch_image(to_8b_image(image))


def tile_images(images, imgs_per_row=4):
    rows = []
    row = []
    imgs_per_row = min(len(images), imgs_per_row)
    for i in range(len(images)):
        row.append(images[i])
        if len(row) == imgs_per_row:
            rows.append(np.concatenate(row, axis=1))
            row = []
    if len(rows) > 2 and len(rows[-1]) != len(rows[-2]):
        rows.pop()
    imgout = np.concatenate(rows, axis=0)
    return imgout

     
class ImageWriter():
    def __init__(self, output_dir, exp_name):
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, exp_name)
        self.obj_dir = os.path.join(output_dir, exp_name+'_3d')
        self.depth_dir = os.path.join(output_dir, exp_name+'_depth')
        print("The rendering is saved in " + \
              colored(self.image_dir, 'cyan'))
        
        # # remove image dir if it exists
        # if os.path.exists(self.image_dir):
        #     shutil.rmtree(self.image_dir)

        # if os.path.exists(self.obj_dir):
        #     shutil.rmtree(self.obj_dir)       
        
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.obj_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        self.frame_idx = -1
        self.name_3d_together = {}
        self.images_np, self.image_names = [], []

    def append(self, image, img_name=None):
        if img_name in self.image_names:
            return
        self.frame_idx += 1
        if img_name is None:
            img_name = f"{self.frame_idx:06d}"
        save_image(image, f'{self.image_dir}/{img_name}.png')
        self.images_np.append(image)
        self.image_names.append(img_name)
        return self.frame_idx, img_name
    
    def append_depth(self, image, img_name=None, to_int32=False):
        if img_name is None:
            img_name = f"{self.frame_idx:06d}_depth"    
        if img_name in self.image_names:
            return
        if to_int32:
            image = image.astype(np.int32)
        output_name = os.path.join(self.depth_dir, img_name+'.npy')
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        np.save(output_name, image)
        return

    def append_3d(self, point3d, mask, obj_name=None, weight_img=None, depth_img=None):
        if obj_name is None:
            obj_name = f"{self.frame_idx:06d}"
        with open(os.path.join(self.obj_dir, f'{obj_name}.obj'),'w') as f:
            us, vs = mask.nonzero()
            for u,v in zip(us,vs):
                xyz = point3d[u,v]
                f.writelines(f'v {xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f}\n')
        if not weight_img is None:
            np.save(os.path.join(self.obj_dir,f'{obj_name}-weights.npy'),weight_img)
        if not depth_img is None:
            np.save(os.path.join(self.obj_dir,f'{obj_name}-depth.npy'),depth_img)
        return
    
    def append_3d_together(self, name, data):
        self.name_3d_together[name] = data

    def append_cnl_3d(self, cnl_xyz, cnl_rgb, obj_name=None):
        if obj_name is None:
            obj_name = f"{self.frame_idx:06d}-cnl"
        with open(os.path.join(self.obj_dir, f'{obj_name}.obj'),'w') as f:
            for xyz,rgb in zip(cnl_xyz,cnl_rgb):
                f.writelines(f'v {xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f} ')
                f.writelines(f'{rgb[0]:.7f} {rgb[1]:.7f} {rgb[2]:.7f} \n')
        return

    def save_pkl(self, obj, name):
        with open(os.path.join(self.obj_dir, name), 'wb') as f:
            pickle.dump(obj, f)

    def finalize(self, video_name=None):
        if self.name_3d_together!={}:
            filename = os.path.join(self.output_dir,'name-2-3d.bin')
            torch.save(self.name_3d_together, filename)
            print('Save as ', filename)

        #save_video
        sorted_idx = sorted(np.arange(len(self.images_np)), key=lambda i:self.image_names[i])
        ref_view = self.image_names[0].split('_')[-1]
        
        image_stack = np.stack([self.images_np[i] for i in sorted_idx if self.image_names[i].split('_')[-1] == ref_view], axis=0)

        if video_name is None:
            video_name = self.image_dir+'.mp4' if self.image_dir[-1]!='/' else self.image_dir[:-1]+'.mp4'
        else:
            video_name = os.path.join(self.image_dir, video_name)
        imageio.mimwrite(video_name, image_stack, format='mp4', fps=10, quality=8)
        pass
