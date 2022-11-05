import torch.nn as nn
from .transformer_tools import DropPath, to_2tuple


class Mlp(nn.Module):
  """MLP as implemented in timm
  """
  def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    drops = to_2tuple(drop)

    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = act_layer()
    self.drop1 = nn.Dropout(drops[0])
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop2 = nn.Dropout(drops[1])

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x


class Attention(nn.Module):
  """Self Attention as implemented in timm
  """
  def __init__(self, d_model, nhead=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
    super().__init__()
    assert d_model % nhead == 0, 'd_model needs to be divisible by nhead'
    self.nhead = nhead
    self.scale = (d_model // nhead) ** -0.5
    

    self.to_qkv = nn.Linear(d_model, d_model*3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(d_model, d_model)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x):
    B, N, C = x.size()
    qkv = self.to_qkv(x).reshape(B, N, 3, self.nhead, C // self.nhead).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    attn = (q @ k.transpose(-1, -2)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x


class Attention_Cross(nn.Module):
  """Attention for decoder layer.Some palce may be called "inter attention"
  """
  def __init__(self, d_model, nhead=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
    super().__init__()
    assert d_model % nhead == 0, 'd_model needs to be divisible by nhead'
    self.nhead = nhead
    self.scale = (d_model // nhead) ** -0.5
    
    self.to_q = nn.Linear(d_model, d_model, bias=qkv_bias)
    self.to_kv = nn.Linear(d_model, d_model*2, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(d_model, d_model)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x, y):
    """
      Args:
        x: output of the former layer
        y: memery of the encoder layer
    """
    B, Nx, C = x.size()
    _, Ny, _ = y.size()
    q = self.to_q(x).reshape(B, Nx, self.nhead, C // self.nhead).permute(0, 2, 1, 3)
    kv = self.to_kv(y).reshape(B, Ny, 2, self.nhead, C // self.nhead).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)

    attn = (q @ k.transpose(-1, -2)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x


class TransformerEncoderLayer(nn.Module):
  """Implemented as vit block in timm
  """
  def __init__(self, d_model, nhead=8, mlp_ratio=4, qkv_bias=False, attn_drop=0., 
         drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=False):
    super().__init__()
    mlp_hidden_dim = int(d_model * mlp_ratio)

    self.attn = Attention(d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)    
    self.mlp = Mlp(d_model, hidden_features=mlp_hidden_dim, out_features=d_model, act_layer=act_layer, drop=drop)
    
    self.norm_first = norm_first
    self.norm1 = norm_layer(d_model)
    self.norm2 = norm_layer(d_model)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    

  def forward(self, x):
    if self.norm_first == True:
      x = x + self.drop_path(self.attn(self.norm1(x)))
      x = x + self.drop_path(self.mlp(self.norm2(x)))
    else:
      x = self.norm1(x + self.drop_path(self.attn(x)))
      x = self.norm2(x + self.drop_path(self.mlp(x)))
    return x


class TransformerDecoderLayer(nn.Module):
  """Transformer Decoder Layer
  """
  def __init__(self, d_model, nhead=8, mlp_ratio=4, qkv_bias=False, attn_drop=0., 
         drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=False):
    super().__init__()
    mlp_hidden_dim = int(d_model * mlp_ratio)
    
    self.attn1 = Attention(d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    self.attn2 = Attention_Cross(d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    self.mlp = Mlp(d_model, hidden_features=mlp_hidden_dim, out_features=d_model, act_layer=act_layer, drop=drop)
    
    self.norm_first = norm_first
    self.norm1 = norm_layer(d_model)
    self.norm2 = norm_layer(d_model)
    self.norm3 = norm_layer(d_model)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    

  def forward(self, x, y):
    """
      Args:
        x: output of the former layer
        y: memery of the encoder layer
    """
    if self.norm_first == True:
      x = x + self.drop_path(self.attn1(self.norm1(x)))
      x = x + self.drop_path(self.attn2(self.norm2(x), y))
      x = x + self.drop_path(self.mlp(self.norm3(x)))
    else:
      x = self.norm1(x + self.drop_path(self.attn1(x)))
      x = self.norm2(x + self.drop_path(self.attn2(x, y)))
      x = self.norm3(x + self.drop_path(self.mlp(x)))
    return x


# Example Decoder_Layer
# import torch
# model = TransformerDecoderLayer(768, nhead=8, norm_first=True)
# tgt = torch.randn(1, 256, 768)
# memory = torch.randn(1, 196, 768)
# output = model(tgt, memory)
# output.shape