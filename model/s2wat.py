import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .transformer_tools import to_2tuple, to_ntuple
from .transformer_components import Mlp, Attention


########################################## Extension components ##########################################

def split_int(num):
  """Split an integer into 2 integers evenly
  Args:
    num (int): The input integer
  Returns:
    num_1 (int)
    num_2 (int)
  """
  if num % 2 == 0:
    num_1 = num_2 = num // 2
  else:
    num_1 = num // 2
    num_2 = num_1 + 1
  return num_1, num_2


def unpad2D(input, pad):
  """Crop the input tensor according to pad.(Inverse operation for padding)
  Args:
    input (Tensor): (B, C, H, W)
    pad (Tuple of int): (left, right, top, bottom)
  Returns:
    output (Tensor): (B, C, new_H, new_W)
  """
  pad_W_left, pad_W_right, pad_H_top, pad_H_bottom = pad
  if pad_H_top == 0 and pad_H_bottom == 0 and not (pad_W_left == 0 and pad_W_right == 0):
    output = input[:, :, :, pad_W_left:-pad_W_right]
  elif pad_W_left == 0 and pad_W_right == 0 and not (pad_H_top == 0 and pad_H_bottom == 0):
    output = input[:, :, pad_H_top:-pad_H_bottom, :]
  elif pad_H_top == 0 and pad_H_bottom == 0 and pad_W_left == 0 and pad_W_right == 0:
    output = input
  else:
    output = input[:, :, pad_H_top:-pad_H_bottom, pad_W_left:-pad_W_right]
  return output


def seq_padding(x, dividable_size, input_resolution, pad_mode='constant'):
  """Padding for sequential data
  Args:
    x (Tensor): (B, L, C)
    dividable_size (Tuple | int): dividable size
    input_resolution (Tuple): resolution of x
  Returns:
    x (Tensor): (B, new_L, C)
    output_resolution (Tuple): new resolution of x
    pad (Tuple of int): (left, right, top, bottom)
  """ 
  H, W = input_resolution
  B, L, C = x.shape
  assert L == H * W, 'Input of wrong size.'
  dividable_size = to_2tuple(dividable_size)
  x = x.permute(0, 2, 1).reshape(B, C, H, W)

  rema_H, rema_W = H % dividable_size[0], W % dividable_size[1]
  pad_H, pad_W = dividable_size[0] - rema_H, dividable_size[1] - rema_W

  pad_H_top, pad_H_bottom = split_int(pad_H) if rema_H != 0 else (0, 0)
  pad_W_left, pad_W_right = split_int(pad_W) if rema_W != 0 else (0, 0)

  x = F.pad(x, (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom), pad_mode, 0)

  padded_H, padded_W = x.shape[-2:]
  x = x.reshape(B, C, -1).permute(0, 2, 1)
  return x, (padded_H, padded_W), (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)


# Example
# x = torch.randn(1, 14, 768)
# y = seq_padding(x, dividable_size=7, input_resolution=(2, 7))
# print(y[0].shape, y[1], y[2])


def seq_unpad(x, input_resolution, pad):
  """Unpadding for sequential data
  Args:
    x (Tensor): (B, L, C)
    input_resolution (Tuple): resolution of x
    pad (Tuple of int): (left, right, top, bottom)
  Returns:
    x (Tensor): (B, new_L, C)
    output_resolution (Tuple): new resolution of x
  """ 
  padded_H, padded_W = input_resolution
  B, L, C = x.shape
  assert L == padded_H * padded_W, 'Input of wrong size.'
  x = x.permute(0, 2, 1).reshape(B, C, padded_H, padded_W)

  x = unpad2D(x, pad=pad)

  H, W = x.shape[-2:]
  x = x.reshape(B, C, -1).permute(0, 2, 1)
  return x, (H, W)


# Example
# x = torch.randn(1, 48, 768)
# y = seq_padding(x, dividable_size=7, input_resolution=(6, 8), pad_mode='constant')
# print(y[0].shape, y[1], y[2])
# z = seq_unpad(y[0], y[1], y[2])
# print(z[0].shape, z[1])


########################################## Basic components ##########################################

def window_partition(x, window_size):
  """Slightly modified for arbitrary window_size & resolution combination
  Args:
    x: (B,H,W,C)
    window_size (tuple[int] | int): window size
  Returns:
    windows: (num_windows*B, window_size, window_size, C)
  """
  window_size = to_2tuple(window_size)
  B, H, W, C = x.shape
  n_win_H = H//window_size[0]
  n_win_W = W//window_size[1]
  if not (H % window_size[0] == 0 and W  % window_size[1] == 0):
    x = x[:, :n_win_H*window_size[0], :n_win_W*window_size[1], :]
  x = x.view(B, n_win_H, window_size[0], n_win_W, window_size[1], C)
  windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
  return windows


def window_reverse(windows, window_size, H, W):
  """Slightly modified for arbitrary window_size & resolution combination
  Args:
    windows: (num_windows*B, window_size, window_size, C)
    window_size (tuple[int] | int): Window size
    H (int): Height of image
    W (int): Width of image
  Returns:
    x: (B, H, W, C)
  """
  window_size = to_2tuple(window_size)
  n_win_H = H//window_size[0]
  n_win_W = W//window_size[1]
  B = windows.shape[0] // (n_win_H*n_win_W)
  x = windows.view(B, n_win_H, n_win_W, window_size[0], window_size[1], -1)
  x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, n_win_H*window_size[0], n_win_W*window_size[1], -1)
  return x


# Example
# H, W = 6, 6
# window_size = 2
# x = torch.randn(1, H, W, 3)
# win = window_partition(x, window_size=window_size)
# y = window_reverse(win, window_size=window_size, H=H, W=W)
# print(x.shape)
# print(win.shape)
# print(y.shape)
# print(x[0].permute(2,0,1)[0])
# print(y[0].permute(2,0,1)[0])


def seq_crop(x, dividable_size, input_resolution):
  """
  Arg:
    x (Tensor): (B, L, C)
    dividable_size (Tuple | int): dividable size
    input_resolution (Tuple): resolution of x
  Returns:
    x (Tensor): (B, new_L, C)
    output_resolution (Tuple): new resolution of x
  """ 
  H, W = input_resolution
  B, L, C = x.shape
  assert L == H * W, 'Input of wrong size.'
  dividable_size = to_2tuple(dividable_size)
  x = x.reshape(B, H, W, C)

  rema_H, rema_W = H % dividable_size[0], W % dividable_size[1]
  new_H, new_W = H - rema_H, W - rema_W
  if rema_H !=0 or rema_W !=0:
    x = x[:, :new_H, :new_W, :]

  x = x.reshape(B, -1, C)
  return x, (new_H, new_W)


# Example
# H, W = 22, 34
# x = torch.randn(2, H*W, 96)
# x, new_size = seq_crop(x, dividable_size=7, input_resolution=(H, W))
# print(x.shape, new_size)


class PatchEmbed_Kai(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.flatten = flatten

        self.proj = nn.Conv2d(self.in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert C==self.in_chans, 'Input image need to have same numbers of channels with the initialed.'
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, (H, W)


# Example
# model = PatchEmbed_Kai(patch_size=4, in_chans=3, embed_dim=96)
# x = torch.randn(1, 3, 224, 224)
# y = model(x)
# torch.save(model.state_dict(), "./PatchEmbed.pkl")
# print(y[0].shape, y[1])


class PatchMerging_Kai(nn.Module):
  """ Patch Merging Layer.
  Args:
    input_resolution (tuple[int] | int): Resolution of input feature.
    d_model (int): Number of input channels.
    norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
  """

  def __init__(self, input_resolution, d_model, norm_layer=nn.LayerNorm):
    super().__init__()
    self.input_resolution = to_2tuple(input_resolution)
    self.d_model = d_model
    self.reduction = nn.Linear(4*d_model, 2*d_model, bias=False)
    self.norm = norm_layer(4 * d_model)

  def forward(self, x):
    """
    Args:
      x (Tuple): (Tensor, arbitrary_input, (H,W)), arbitrary_input (bool)
        if arbitrary_input=False, (H,W) will not be required
        B, H*W, C -> B, H/2*W/2, 4*C
    """
    arbitrary_input = x[1]
    if arbitrary_input:
      H, W = x[2]
    else:
      H, W = self.input_resolution
      
    x = x[0]
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    x = x.view(B, H, W, C)
    if H % 2 != 0:
      x = x[:, 0:-1, :, :]
    if W % 2 != 0:
      x = x[:, :, 0:-1, :]

    x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
    H, W = x.shape[1], x.shape[2]
    x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

    x = self.norm(x)
    x = self.reduction(x)

    return x, arbitrary_input, (H, W)


# Example
# model = PatchMerging_Kai(input_resolution=(5,4), d_model=3)
# arbitrary_input = True
# # x = torch.randn(1, 20, 3)
# # y = model((x, arbitrary_input))
# x = torch.randn(1, 45, 3)
# y = model((x, arbitrary_input, (5,9)))
# print(y[0].shape, y[2])


class WindowAttention_Kai(nn.Module):
  def __init__(self, d_model, window_size, nhead, qkv_bias=True, attn_drop=0., proj_drop=0.):
    super().__init__()
    assert d_model % nhead == 0, 'd_model needs to be divisible by nhead'
    self.window_size = to_2tuple(window_size)
    self.nhead = nhead
    self.scale = (d_model // nhead) ** -0.5

    # Relative Position Bias's parameter Table
    self.relative_position_bias_table = nn.Parameter(
      torch.zeros((2*self.window_size[0]-1)*(2*self.window_size[1]-1), nhead)
    )

    # Compute indice of relative_position_bias_table for attention matrix
    coords_h = torch.arange(self.window_size[0])
    coords_w = torch.arange(self.window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += self.window_size[0] - 1
    relative_coords[:, :, 1] += self.window_size[1] - 1
    relative_coords[:, :, 0] *= 2*self.window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)
    self.register_buffer("relative_position_index", relative_position_index)

    self.qkv = nn.Linear(d_model, d_model*3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(d_model, d_model)
    self.proj_drop = nn.Dropout(proj_drop)

    nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
  
  def forward(self, x, shape):
    H, W = shape
    Bi, Ni, Ci = x.size()
    assert Ni == H * W, "Inputs with wrong size."
    x = x.reshape(Bi, H, W, Ci)

    # print(self.window_size)
    x = window_partition(x, self.window_size)
    x = x.reshape(-1, self.window_size[0]*self.window_size[1], Ci)

    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.nhead, C//self.nhead).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    attn = (q @ k.transpose(-2, -1)) * self.scale

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
      self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1], self.nhead)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    attn = attn + relative_position_bias.unsqueeze(0)

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    x = window_reverse(x, self.window_size, H, W)
    x = x.reshape(Bi, Ni, Ci)
    return x 

# Example
# model = WindowAttention_Kai(
#   d_model=768,
#   window_size=7,
#   nhead=8
# )
# x = torch.randn(1, 784, 768)
# y = model(x, shape=(28, 28))
# print(y.shape)


class StripAttention(nn.Module):
  def __init__(self, d_model, nhead=8, strip_width=7, is_vertical=False, qkv_bias=False, attn_drop=0., proj_drop=0.):
    super().__init__()
    self.d_model = d_model
    self.strip_width = strip_width
    self.is_vertical = is_vertical

    self.attn = Attention(
      d_model=d_model,
      nhead=nhead,
      qkv_bias=qkv_bias,
      attn_drop=attn_drop,
      proj_drop=proj_drop,
    )

  def forward(self, x, shape):
    H, W = shape
    B, N, C = x.size()
    assert N == H * W, "Inputs with wrong size."
    x = x.reshape(B, H, W, C)

    # print(self.strip_width)
    if self.is_vertical:
      x = window_partition(x, (H, self.strip_width))
      x = x.reshape(-1, H*self.strip_width, C)
    else:
      x = window_partition(x, (self.strip_width, W))
      x = x.reshape(-1, W*self.strip_width, C)
    
    wins = self.attn(x)

    if self.is_vertical:
      x = window_reverse(wins, (H, self.strip_width), H, W)
    else:
      x = window_reverse(wins, (self.strip_width, W), H, W)

    x = x.reshape(B, N, C)
    return x

# Example
# model = StripAttention(
#   d_model=768,
#   nhead=8,
#   strip_width=7,
#   is_vertical=False
# )
# x = torch.randn(1, 784, 768)
# y = model(x, (28, 28))
# print(y.shape)


class StripAttentionBlock(nn.Module):
  def __init__(self, d_model, input_resolution, nhead=8, strip_width=7,
         mlp_ratio=4, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
         act_layer=nn.GELU, norm_layer=nn.LayerNorm):
    super().__init__()
    self.d_model = d_model
    self.input_resolution = to_2tuple(input_resolution)
    self.strip_width = strip_width

    self.norm1 = norm_layer(d_model)

    self.attn1 = StripAttention(
      d_model=d_model,
      nhead=nhead,
      strip_width=strip_width,
      is_vertical=False,
      qkv_bias=qkv_bias,
      attn_drop=attn_drop,
      proj_drop=drop
    )
    self.attn2 = StripAttention(
      d_model=d_model,
      nhead=nhead,
      strip_width=strip_width,
      is_vertical=True,
      qkv_bias=qkv_bias,
      attn_drop=attn_drop,
      proj_drop=drop
    )
    self.attn3 = WindowAttention_Kai(
      d_model=d_model,
      window_size=(strip_width*2, strip_width*2),
      nhead=nhead,
      qkv_bias=qkv_bias,
      attn_drop=attn_drop,
      proj_drop=drop
    )

    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    self.norm2 = norm_layer(d_model)
    mlp_hidden_dim = int(d_model * mlp_ratio)
    self.mlp = Mlp(d_model, hidden_features=mlp_hidden_dim, out_features=d_model, act_layer=act_layer, drop=drop)


  def forward(self, x):
    arbitrary_input = x[1]
    if arbitrary_input:
      H, W = x[2]
      # x, (H, W) = seq_crop(x[0], dividable_size=self.strip_width*2, input_resolution=(H, W))
      x, (H, W), pad = seq_padding(x[0], dividable_size=self.strip_width*2, input_resolution=(H, W), pad_mode='constant')
    else:
      H, W = self.input_resolution
      x = x[0]

    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x
    x = self.norm1(x)

    x1 = self.attn1(x, shape=(H,W))
    x2 = self.attn2(x, shape=(H,W))
    x3 = self.attn3(x, shape=(H,W))

    # Method 1
    # x = x1 + x2 + x3
    # Method 2
    q_x = x.unsqueeze(dim=2)
    k_x = torch.stack([x, x1, x2, x3], dim=2)
    attn_x = (q_x @ k_x.transpose(-1, -2)).softmax(dim=-1)
    x = attn_x @ k_x
    x = x.squeeze(dim=2)
    x = shortcut + self.drop_path(x)

    # FFN
    x = x + self.drop_path(self.mlp(self.norm2(x)))

    if arbitrary_input:
      x, (H, W) = seq_unpad(x, (H, W), pad)

    return (x, arbitrary_input, (H,W))


# Example
# model = StripAttentionBlock(
#   d_model=96,
#   input_resolution=28,
#   nhead=8,
#   strip_width=7
# )
# arbitrary_input = True
# # x = (torch.randn(1, 784, 96), arbitrary_input, (28,28))
# # y = model(x)
# x = (torch.randn(1, 840, 96), arbitrary_input, (28, 30))
# y = model(x)
# print(y[0].shape, y[2])


class BasicLayer_SA(nn.Module):
  def __init__(self, d_model, input_resolution, depth, nhead, strip_width,
         mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
         drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
    super().__init__()
    self.d_model = d_model
    self.input_resolution = to_2tuple(input_resolution)
    self.depth = depth
    self.strip_width = list(to_ntuple(self.depth)(strip_width))
    self.use_checkpoint = use_checkpoint

    # build blocks
    self.blocks = nn.ModuleList([
      StripAttentionBlock(
        d_model=d_model,
        input_resolution=self.input_resolution,
        nhead=nhead,
        strip_width=self.strip_width[i],
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop=drop,
        attn_drop=attn_drop,
        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        norm_layer=norm_layer
      )
      for i in range(self.depth)
    ])

    # patch merging layer
    if downsample is not None:
      self.downsample = downsample(self.input_resolution, d_model=d_model, norm_layer=norm_layer)
    else:
      self.downsample = None


  def forward(self, x):
    for blk in self.blocks:
      if self.use_checkpoint:
        x = checkpoint.checkpoint(blk, x)
      else:
        x = blk(x)
    # print(x[0].shape, x[1], x[2])
    if self.downsample is not None:
      x = self.downsample(x)
    return x


# Example
# model = BasicLayer_SA(
#   d_model=768,
#   input_resolution=112,
#   depth=3,
#   nhead=8,
#   strip_width=[7, 2, 7],
#   drop_path=0.,
#   downsample=PatchMerging_Kai,
#   use_checkpoint=False
# )
# arbitrary_input = True
# # x = (torch.randn(1, 12544, 768), arbitrary_input, (112,112))
# # y = model(x)
# x = (torch.randn(1, 810, 768), arbitrary_input, (27,30))
# y = model(x)
# print(y[0].shape, y[2])


########################################## S2WAT ##########################################

class S2WAT(nn.Module):
  def __init__(self, img_size=224, patch_size=4, in_chans=3,
         embed_dim=96, depths=[2, 2, 6, 2], nhead=[3, 6, 12, 24],
         strip_width=7, mlp_ratio=4., qkv_bias=True,
         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
         use_checkpoint=False):
    super().__init__()

    self.img_size = to_2tuple(img_size)
    self.patch_size = to_2tuple(patch_size)
    self.num_layers = len(depths)
    self.strip_width = list(to_ntuple(self.num_layers)(strip_width))
    self.embed_dim = embed_dim
    self.ape = ape
    self.patch_norm = patch_norm

    # split image into non-overlapping patches
    self.patch_embed = PatchEmbed_Kai(
      patch_size=patch_size,
      in_chans=in_chans,
      embed_dim=embed_dim,
      norm_layer=norm_layer if self.patch_norm else None
    )
    self.patches_resolution = (self.img_size[0]//self.patch_size[0], self.img_size[1]//self.patch_size[1])
    self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

    # absolute position embedding
    if self.ape:
      self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
      nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
    self.pos_drop = nn.Dropout(drop_rate)

    # stochastic depth
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

    # build layers
    self.layers = nn.ModuleList()
    for i in range(self.num_layers):
      layer = BasicLayer_SA(
        d_model=int(self.embed_dim * 2 ** i),
        input_resolution=(self.patches_resolution[0] // (2**i),
                 self.patches_resolution[1] // (2**i)),
        depth=depths[i],
        nhead=nhead[i],
        strip_width=self.strip_width[i],
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop=drop_rate,
        attn_drop=attn_drop_rate,
        drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
        norm_layer=norm_layer,
        downsample=PatchMerging_Kai if (i < self.num_layers - 1) else None,
        use_checkpoint=use_checkpoint
      )
      self.layers.append(layer)

    self.apply(self._init_weights)

  def _init_weights(self,m):
    if isinstance(m, nn.Linear):
      nn.init.trunc_normal_(m.weight, std=.02)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)

  @torch.jit.ignore
  def no_weight_decay(self):
    return {'absolute_pos_embed'}

  @torch.jit.ignore
  def no_weight_decay_keywords(self):
    return {'relative_position_bias_table'}

  def forward_features(self, x):
    x, arbitrary_input = x[0], x[1]
    x, (H, W) = self.patch_embed(x)
    if self.ape:
      x = x + self.absolute_pos_embed
    x = self.pos_drop(x)

    x = (x, arbitrary_input, (H, W))
    for layer in self.layers:
      x = layer(x)
      # print(x[0].shape)

    return x

  def forward(self, x, arbitrary_input=False):
    if arbitrary_input:
      H, W = x.shape[2], x.shape[3]
      x = (x, arbitrary_input, (H, W))
    else:
      x = (x, arbitrary_input)

    x = self.forward_features(x)
    return x


# Example
# model = S2WAT(
#   img_size=224,
#   patch_size=2,
#   in_chans=3,
#   embed_dim=96,
#   depths=[2, 2, 2],
#   nhead=[6, 12, 24],
#   strip_width=[2, 4, [2,7]],
#   drop_path_rate=0.,
#   patch_norm=True
# )
# # x = torch.randn(1, 3, 224, 224)
# x = torch.randn(1, 3, 464, 376)
# y = model(x, arbitrary_input=True)
# print(y[0].shape, y[2])