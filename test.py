import argparse
import torch
import torch.nn as nn
from pathlib import Path

from model.configuration import TransModule_Config
from model.s2wat import S2WAT
from net import TransModule, Decoder_MVGG
from tools import save_transferred_imgs, Sample_Test_Net


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--input_dir', type=str, default='./input/Test',
                    help='Directory path to a batch of content and style images ' + \
                    'which are loaded in "Content"/"Style" subfolders respectively.')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Directory to save the output image(s)')
parser.add_argument('--checkpoint_import_path', type=str, default='./pre_trained_models/checkpoint/checkpoint_40000_epoch.pkl',
                    help='Directory path to the importing checkpoint')

args = parser.parse_args()

# Print args
print('Running args: ')
for k, v in sorted(vars(args).items()):
    print(k, '=', v)
print()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)


# Models Config
transModule_config = TransModule_Config(
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
            norm_first=True
            )

# Hardware Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
# Models
encoder = S2WAT(
  img_size=224,
  patch_size=2,
  in_chans=3,
  embed_dim=192,
  depths=[2, 2, 2],
  nhead=[3, 6, 12],
  strip_width=[2, 4, 7],
  drop_path_rate=0.,
  patch_norm=True
)
decoder = Decoder_MVGG(d_model=768, seq_input=True)
transModule = TransModule(transModule_config)

network = Sample_Test_Net(encoder, decoder, transModule)

# Load the checkpoint
print('loading checkpoint...')
checkpoint = torch.load(args.checkpoint_import_path, map_location=device)

network.encoder.load_state_dict(checkpoint['encoder'])
network.decoder.load_state_dict(checkpoint['decoder'])
network.transModule.load_state_dict(checkpoint['transModule'])

loss_count_interval = checkpoint['loss_count_interval']
print('loading finished')

# Load the model to device 
network.to(device)

# ===============================================Test===============================================

# Arbitrary size inputs transfer (Single picture)
save_transferred_imgs(network, args.input_dir, args.output_dir, device=device)