import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from tqdm import tqdm


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, ii, output_file):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    mask = np.sum(np.abs(img), axis=2) > 0
    flo[mask==False] = 0
    flo = flow_viz.flow_to_image(flo)
    flo = flo * mask[:,:,None]
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    #cv2.waitKey()
    cv2.imwrite(output_file, img_flo[:, :, [2,1,0]])


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = sorted(glob.glob(args.path))
        
        images = sorted(images)
        for ii,(imfile1, imfile2) in tqdm(enumerate(zip(images[:-1], images[1:]))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            output_file = os.path.join(args.output_path, os.path.basename(imfile1))
            viz(image1, flow_up,ii, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--output_path', help="path to save visualization")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    demo(args)
