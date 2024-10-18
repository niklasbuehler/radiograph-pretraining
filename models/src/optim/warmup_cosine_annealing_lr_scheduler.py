import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, max_steps=None, warmup_steps=0, eta_min=0, last_epoch=-1):
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr = [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
            #print("Step", self.last_epoch, "(warmup). lr:", lr)
            return lr
        else:
            # Cosine annealing
            lr = [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi *
                        (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                    ))
                    for base_lr in self.base_lrs
            ]
            #print("Step", self.last_epoch, "(annealing). lr:", lr)
            return lr