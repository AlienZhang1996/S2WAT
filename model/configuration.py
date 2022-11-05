import torch.nn as nn

class TransModule_Config():
  def __init__(
    self,
    nlayer=3,
    d_model=768,
    nhead=8,
    mlp_ratio=4,
    qkv_bias=False,
    attn_drop=0.,
    drop=0.,
    drop_path=0.,
    act_layer=nn.GELU,
    norm_layer=nn.LayerNorm,
    norm_first=False
  ):
    self.nlayer = nlayer
    self.d_model = d_model
    self.nhead = nhead
    self.mlp_ratio = mlp_ratio
    self.qkv_bias = qkv_bias
    self.attn_drop = attn_drop
    self.drop = drop
    self.drop_path = drop_path
    self.act_layer = act_layer
    self.norm_layer = norm_layer
    self.norm_first = norm_first