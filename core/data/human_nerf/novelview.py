from core.data.human_nerf import train
import os, cv2, pickle, numpy as np
from core.utils.file_util import list_files, split_path
from core.utils.image_util import load_image
from configs import cfg

class Dataset(train.Dataset):
    def __init__(self, novelviews, **kwargs):
        self.novelviews = novelviews
        super().__init__(**kwargs)
    
    def load_train_cameras():
        return #?


    def get_freeview_camera(self, frame_idx, **kwargs):
        return K, E