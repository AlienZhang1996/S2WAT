import math
import numpy as np
from torch.optim import Optimizer, lr_scheduler


class WarmUpLR(lr_scheduler._LRScheduler):
  """Warm Up learning rate strategy

  Before warmup_step, lr increases from 0 to initial learning rate gradually
  After warmup_step, lr decreases from initial learning rate according to the negative square root of "step"
  If warmup_step=0, skip the phase of warm up and step into decay phase directly

  """
  def __init__(self, optimizer, warmup_step, last_epoch=-1):
    self.warmup_step = warmup_step
    super(WarmUpLR, self).__init__(optimizer, last_epoch=last_epoch)
    

  def get_lr(self):
    if self.last_epoch < self.warmup_step:
      return [base_lr * ((self.last_epoch + 1) / self.warmup_step) \
          for base_lr in self.base_lrs]
    else:
      return [base_lr * math.pow(self.last_epoch + 1, -0.5) \
          for base_lr in self.base_lrs]


class CosineAnnealingWarmUpLR(lr_scheduler._LRScheduler):
  """Warm Up learning rate strategy

  Before warmup_step, lr increases from 0 to initial learning rate of "optimizer" gradually
  After warmup_step, lr decreases from initial learning rate according to cosine annealing
  If warmup_step=0, skip the phase of warm up and step into decay phase directly

  """
  def __init__(self, optimizer, warmup_step, max_step, min_lr=0, last_epoch=-1):
    self.warmup_step = warmup_step
    self.max_step = max_step
    self.min_lr = min_lr
    super(CosineAnnealingWarmUpLR, self).__init__(optimizer, last_epoch=last_epoch)
    

  def get_lr(self):
    if self.last_epoch < self.warmup_step:
      return [base_lr * ((self.last_epoch + 1) / self.warmup_step) \
          for base_lr in self.base_lrs]
    else:
      return [self.min_lr + 0.5*(base_lr-self.min_lr) * \
          (1 + np.cos(np.pi * (self.last_epoch-self.warmup_step)/(self.max_step-self.warmup_step))) \
          for base_lr in self.base_lrs]


def adjust_learning_rate(optimizer, iteration_count, lr, lr_decay):
    """Imitating the original implementation"""
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr