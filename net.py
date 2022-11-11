import torch.nn as nn
from model.configuration import TransModule_Config
from model.transformer_components import TransformerDecoderLayer


########################################## VGG & components ##########################################

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


# compute channel-wise means and variances of features
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4, 'The shape of feature needs to be a tuple with length 4.'
    B, C = size[:2]
    feat_mean = feat.reshape(B, C, -1).mean(dim=2).reshape(B, C, 1, 1)
    feat_std = (feat.reshape(B, C, -1).var(dim=2) + eps).sqrt().reshape(B, C, 1, 1)
    return feat_mean, feat_std


# normalize features
def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


########################################## Transfer Module ##########################################

class TransModule(nn.Module):
  """The Transfer Module of Style Transfer via Transformer

  Taking Transformer Decoder as the transfer module.

  Args:
    config: The configuration of the transfer module
  """
  def __init__(self, config: TransModule_Config=None):
    super(TransModule, self).__init__()
    self.layers = nn.ModuleList([
      TransformerDecoderLayer(
          d_model=config.d_model,
          nhead=config.nhead,
          mlp_ratio=config.mlp_ratio,
          qkv_bias=config.qkv_bias,
          attn_drop=config.attn_drop,
          drop=config.drop,
          drop_path=config.drop_path,
          act_layer=config.act_layer,
          norm_layer=config.norm_layer,
          norm_first=config.norm_first
          ) \
      for i in range(config.nlayer)
    ])

  def forward(self, content_feature, style_feature):
    """
    Args:
      content_feature: Content features，for producing Q sequences. Similar to tgt sequences in pytorch. (Tensor,[Batch,sequence,dim])
      style_feature : Style features，for producing K,V sequences.Similar to memory sequences in pytorch.(Tensor,[Batch,sequence,dim])

    Returns:
      Tensor with shape (Batch,sequence,dim)
    """
    for layer in self.layers:
      content_feature = layer(content_feature, style_feature)
    
    return content_feature


# Example
# import torch
# transModule_config = TransModule_Config(
#             nlayer=3,
#             d_model=768,
#             nhead=8,
#             mlp_ratio=4,
#             qkv_bias=False,
#             attn_drop=0.,
#             drop=0.,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm,
#             norm_first=True
#             )
# transModule = TransModule(transModule_config)
# tgt = torch.randn(1, 20, 768)
# memory = torch.randn(1, 10, 768)
# print(transModule(tgt, memory).shape)


########################################## Decoder ##########################################

decoder_stem = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


class Decoder_MVGG(nn.Module):
  def __init__(self, d_model=768, seq_input=False):
      super(Decoder_MVGG, self).__init__()
      self.d_model = d_model
      self.seq_input = seq_input
      self.decoder = nn.Sequential(
        # Proccess Layer 1        

        # Upsample Layer 2
        nn.ReflectionPad2d(1),
        nn.Conv2d(int(self.d_model), 256, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),

        # Upsample Layer 3
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 128, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3, 1, 0),
        nn.ReLU(),

        # Upsample Layer 4
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 64, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3, 1, 0),
        nn.ReLU(),

        # Channel to 3
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 3, 3, 1, 0),
      )
        
        
  def forward(self, x, input_resolution):
    if self.seq_input == True:
      B, N, C = x.size()
#       H, W = math.ceil(self.img_H//self.patch_size), math.ceil(self.img_W//self.patch_size)
      (H, W) = input_resolution
      x = x.permute(0, 2, 1).reshape(B, C, H, W)
    x = self.decoder(x)  
    return x


# Example 1
# import torch
# decoder = Decoder_MVGG(d_model=768, seq_input=True)
# x = torch.randn(1, 3087, 768)
# y = decoder(x, input_resolution=(63, 49))
# print(y.shape)


########################################## Net ##########################################

class Net(nn.Module):
  def __init__(self, encoder, decoder, transModule, lossNet):
    super(Net, self).__init__()
    self.mse_loss = nn.MSELoss()
    self.encoder = encoder
    self.decoder = decoder
    self.transModule = transModule

    # features of intermediate layers
    lossNet_layers = list(lossNet.children())
    self.feat_1 = nn.Sequential(*lossNet_layers[:4])  # input -> relu1_1
    self.feat_2 = nn.Sequential(*lossNet_layers[4:11]) # relu1_1 -> relu2_1
    self.feat_3 = nn.Sequential(*lossNet_layers[11:18]) # relu2_1 -> relu3_1
    self.feat_4 = nn.Sequential(*lossNet_layers[18:31]) # relu3_1 -> relu4_1
    self.feat_5 = nn.Sequential(*lossNet_layers[31:44]) # relu3_1 -> relu4_1

    # fix parameters
    for name in ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5']:
      for param in getattr(self, name).parameters():
        param.requires_grad = False


  # get intermediate features
  def get_interal_feature(self, input):
    result = []
    for i in range(5):
      input = getattr(self, 'feat_{:d}'.format(i+1))(input)
      result.append(input)
    return result
  

  def calc_content_loss(self, input, target, norm = False):
    assert input.size() == target.size(), 'To calculate loss needs the same shape between input and taget.'
    assert target.requires_grad == False, 'To calculate loss target shoud not require grad.'
    if norm == False:
        return self.mse_loss(input, target) 
    else:
        return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))


  def calc_style_loss(self, input, target):
    assert input.size() == target.size(), 'To calculate loss needs the same shape between input and taget.'
    assert target.requires_grad == False, 'To calculate loss target shoud not require grad.'
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return self.mse_loss(input_mean, target_mean) + \
        self.mse_loss(input_std, target_std)


  # calculate losses
  def forward(self, i_c, i_s):
    f_c = self.encoder(i_c)
    f_s = self.encoder(i_s)
    f_c, f_c_reso = f_c[0], f_c[2]
    f_s, f_s_reso = f_s[0], f_s[2]
    
    f_cs = self.transModule(f_c, f_s)
    f_cc = self.transModule(f_c, f_c)
    f_ss = self.transModule(f_s, f_s)
    
    i_cs = self.decoder(f_cs, f_c_reso)
    i_cc = self.decoder(f_cc, f_c_reso)
    i_ss = self.decoder(f_ss, f_c_reso)
    
    f_c_loss = self.get_interal_feature(i_c)
    f_s_loss = self.get_interal_feature(i_s)
    f_i_cs_loss = self.get_interal_feature(i_cs)
    f_i_cc_loss = self.get_interal_feature(i_cc)
    f_i_ss_loss = self.get_interal_feature(i_ss)

    loss_id_1 = self.mse_loss(i_cc, i_c) + self.mse_loss(i_ss, i_s)

    loss_c, loss_s, loss_id_2 = 0, 0, 0
    
    loss_c = self.calc_content_loss(f_i_cs_loss[-2], f_c_loss[-2], norm=True) + \
             self.calc_content_loss(f_i_cs_loss[-1], f_c_loss[-1], norm=True)
    for i in range(1, 5):
      loss_s += self.calc_style_loss(f_i_cs_loss[i], f_s_loss[i])
      loss_id_2 += self.mse_loss(f_i_cc_loss[i], f_c_loss[i]) + self.mse_loss(f_i_ss_loss[i], f_s_loss[i])
    
    return loss_c, loss_s, loss_id_1, loss_id_2, i_cs


# Example 1
# import torch
# from model.s2wat import S2WAT
# transModule_config = TransModule_Config(
#             nlayer=3,
#             d_model=384,
#             nhead=8,
#             mlp_ratio=4,
#             qkv_bias=False,
#             attn_drop=0.,
#             drop=0.,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm,
#             norm_first=True
#             )
# encoder = S2WAT(
#   img_size=224,
#   patch_size=2,
#   in_chans=3,
#   embed_dim=96,
#   depths=[2, 2, 2],
#   nhead=[3, 6, 12],
#   strip_width=[2, 4, 7],
#   drop_path_rate=0.,
#   patch_norm=True
# )
# transModule = TransModule(transModule_config)
# decoder = Decoder_MVGG(d_model=384, seq_input=True)
# vgg.load_state_dict(torch.load('../input/vggpretrainedmodel/vgg_normalised.pth'))
# net = Net(encoder, decoder, transModule, vgg)
# i_c = torch.randn(1, 3, 224, 224)
# i_s = torch.randn(1, 3, 224, 224)
# loss_c, loss_s, loss_id_1, loss_id_2, i_cs = net(i_c, i_s)
# print(loss_c.item(), loss_s.item(), loss_id_1.item(), loss_id_2.item())
# print(i_cs.shape)