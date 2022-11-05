import torch.nn as nn
from itertools import repeat
from typing import Iterable


def _ntuple(n):
    """Copy item to be a tuple with n length (Implemented as timm)
    """
    def parse(x):
      if isinstance(x, Iterable):
        return x
      else:
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
to_ntuple = _ntuple


class DropPath(nn.Module):
    """Stochasticly zero channels of data.(Implemented as timm)
    """
    def __init__(self, drop=0.5, scale=True):
      super().__init__()
      self.drop = drop
      self.scale = scale

    def forward(self, x):
      return self.drop_path(x, self.drop, self.training, self.scale)

    def drop_path(self, x, drop=0.5, training=True, scale=True):
      if drop == 0. or not training:
        return x
      drop_p = 1 - drop
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      random_tensor = x.new_empty(shape).bernoulli_(drop_p)
      if drop_p > 0. and scale:
        random_tensor.div_(drop_p)
    
      return x * random_tensor