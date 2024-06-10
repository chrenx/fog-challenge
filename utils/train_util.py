import os, yaml

import torch
from torch.optim import lr_scheduler


class CustomLRScheduler(lr_scheduler._LRScheduler):
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        initial_lr: float, 
        warmup_steps: int, 
        last_epoch: int = -1
    ):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # override LRScheduler
        return [self.lr_lambda(base_lr) for base_lr in self.base_lrs]

    def lr_lambda(self, step):
        return min(self.initial_lr, self.initial_lr * (step / self.warmup_steps))


def cycle_dataloader(dl):
    while True:
        for data in dl:
            # print(data.shape)
            yield data

def save_group_args(opt):
    with open(os.path.join(opt.save_dir, 'opt.yaml'), 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
        
