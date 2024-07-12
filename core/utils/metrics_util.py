import os, torch, numpy as np
import torch.nn as nn
from third_parties.lpips import LPIPS
from core.utils.network_util import set_requires_grad
from configs import cfg
from skimage.metrics import structural_similarity
from collections import defaultdict
from core.utils.DME_util import DME_Computer
class MetricsWriter(object):
    def __init__(self, output_dir, exp_name, dataset, lpips_computer=None):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.exp_name = exp_name
        self.per_img_f = open(os.path.join(output_dir, f'{exp_name}-metrics.perimg.txt'),'a')
        self.average_f = open(os.path.join(output_dir, f'{exp_name}-metrics.average.txt'),'a')
        self.per_img_f.writelines(f'========={dataset}==========\n')
        self.average_f.writelines(f'========={dataset}==========\n')
        
        self.name2metrics = {}
        self.metrics2ave = defaultdict(int)

        if lpips_computer==None:
            self.lpips_computer = LpipsComputer()
        else:
            self.lpips_computer = lpips_computer
        self.metrics_func = {
            "psnr": lambda pred, target, mask: compute_psnr(pred, target, mask).item(),
            "lpips": lambda pred, target, mask: 1000*self.lpips_computer.compute_lpips(pred=pred.cuda(), target=target.cuda()).item(), 
            "ssim": lambda pred, target, mask: compute_ssim(pred, target, mask).item()
        }
        
        self.dme_computer = DME_Computer()
        self.N = 0

    def normalize(self, img):
        if type(img)==np.ndarray:
            img = torch.tensor(img, dtype=torch.float32)
        if torch.max(img)>2:
            img /= 255
        return img

    def append(self, name, pred, target, mask=None):
        
        if name in self.name2metrics:
            return #already computed
        else:
            self.N += 1
            pred, target = self.normalize(pred), self.normalize(target)
            self.per_img_f.writelines(f'{name}: ')
            self.name2metrics[name] = {}

            for k in cfg.eval.metrics:
                self.name2metrics[name][k] = self.metrics_func[k](pred, target, mask)
                self.metrics2ave[k] += self.name2metrics[name][k]
                self.per_img_f.writelines('{}-{:.4f} '.format(k, self.name2metrics[name][k]))
            
            #DME
            dme = self.dme_computer.append(target, pred, name)
            if dme is not None:
                self.per_img_f.writelines(f'DME-{dme:.4f} ')
                self.name2metrics[name]['dme'] = dme
            self.per_img_f.writelines('\n')   

    def finalize(self):

        self.metrics2ave = {k:v/self.N for k,v in self.metrics2ave.items() if not k == 'dme'}      
        self.metrics2ave['dme'] = self.dme_computer.finalize()
        for k, v in self.metrics2ave.items():
            self.average_f.writelines(f'{k[0]}:{v:.4f}\n') 
        self.per_img_f.close()
        self.average_f.close()

class LpipsComputer(object):
    def __init__(self):
        self.lpips = LPIPS(net='vgg', lpips=cfg.lpips.lpips, layers=cfg.lpips.layers)
        set_requires_grad(self.lpips, requires_grad=False)
        self.lpips = self.lpips.cuda() #nn.DataParallel(self.lpips).cuda()
        return
    def compute_lpips(self, pred, target):
        from core.train.trainers.human_nerf.trainer import scale_for_lpips
        if pred.dim()==3:
            pred, target = pred[None,...], target[None,...]
        with torch.no_grad():
            lpips_loss = self.lpips(scale_for_lpips(pred.permute(0, 3, 1, 2)), 
                                    scale_for_lpips(target.permute(0, 3, 1, 2)))
        return torch.mean(lpips_loss)


def compute_psnr_from_mse(mse):
    return -10.0 * torch.log(mse) / np.log(10.0)

def compute_psnr(pred, target, mask=None):
    """Compute psnr value (we assume the maximum pixel value is 1)."""
    # 0~1
    if mask is not None:
        mask = torch.tile(mask,[1,1,3])
        pred, target = pred[mask], target[mask]
    mse = ((pred - target) ** 2).mean()
    return compute_psnr_from_mse(mse)

def compute_ssim(pred, target, mask=None):
    """Computes Masked SSIM following the neuralbody paper."""
    # 0~1
    assert pred.shape == target.shape #and pred.shape[-1] == 3
    if mask is not None:
        x, y, w, h = cv2.boundingRect(mask.cpu().numpy().astype(np.uint8))
        pred = pred[y : y + h, x : x + w]
        target = target[y : y + h, x : x + w]
    try:
        ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), channel_axis=-1, data_range=1,
        )
    except ValueError:
        ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), multichannel=True, data_range=1,
        )
    return ssim

