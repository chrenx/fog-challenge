import math, os, yaml

import torch
import numpy as np
from torch.optim import lr_scheduler


# class CustomLRScheduler(lr_scheduler._LRScheduler):
#     def __init__(
#         self, 
#         optimizer: torch.optim.Optimizer, 
#         initial_lr: float, 
#         warmup_steps: int, 
#         last_epoch: int = -1
#     ):
#         self.initial_lr = initial_lr
#         self.warmup_steps = warmup_steps
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):  # override LRScheduler
#         return [self.lr_lambda(base_lr) for base_lr in self.base_lrs]

#     def lr_lambda(self, step):
#         return min(self.initial_lr, self.initial_lr * (step / self.warmup_steps))

def norm_axis(a,b,c):
    newa=a/(math.sqrt(float(a*a+b*b+c*c)))
    newb=b/(math.sqrt(float(a*a+b*b+c*c)))
    newc=c/(math.sqrt(float(a*a+b*b+c*c)))
    return ([newa,newb,newc])

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)], 
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)], 
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def rotateC(image,theta,a,b,c): ## theta: angle, a, b, c, eular vector
    axis=norm_axis(a,b,c)
    imagenew=np.dot(image, rotation_matrix(axis,theta))
    return imagenew


def cycle_dataloader(dl):
    while True:
        for data in dl:
            # print(data.shape)
            yield data

def save_group_args(opt):
    with open(os.path.join(opt.save_dir, 'opt.yaml'), 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
        

def count_model_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

