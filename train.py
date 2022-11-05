import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

from torch.utils.data import DataLoader
from model.configuration import TransModule_Config
from model.s2wat import S2WAT
from net import vgg, TransModule, Decoder_MVGG, Net
from dataset_sampler import SimpleDataset, InfiniteSamplerWrapper
from scheduler import CosineAnnealingWarmUpLR
from tools import save_checkpoint


"""Parameters that needs attention
1.epoch           : How many iterations this training has
2.epoch_start        : From which iteration to start the training
3.checkpoint_save_interval : The interval to save checkpoints
4.loss_count_interval    : The interval to calculate the average loss
"""

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg_dir', type=str, default='./pre_trained_models/vgg_normalised.pth')

# training options
parser.add_argument('--base_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epoch', type=int, default=40000)

parser.add_argument('--content_weight', type=float, default=2)
parser.add_argument('--style_weight', type=float, default=3)
parser.add_argument('--id1_weight', type=float, default=50)
parser.add_argument('--id2_weight', type=float, default=1)

# save and count options
parser.add_argument('--checkpoint_save_interval', type=int, default=10000)
parser.add_argument('--loss_count_interval', type=int, default=400)
parser.add_argument('--resume_train', type=bool, default=False, help='Use checkpoints to train or not ')
parser.add_argument('--checkpoint_save_path', type=str, default='./pre_trained_models/checkpoint',
                    help='Directory path to save a checkpoint')
parser.add_argument('--checkpoint_import_path', type=str, default='./pre_trained_model/checkpoint/checkpoint_40000_epoch.pkl',
                    help='Directory path to the importing checkpoint')

args = parser.parse_args()

# Print args
print('Running args: ')
for k, v in sorted(vars(args).items()):
    print(k, '=', v)
print()

if not os.path.exists(args.checkpoint_save_path):
    os.mkdir(args.checkpoint_save_path)

epoch_start = 0
loss_count_interval = args.loss_count_interval


# Model Config
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

# Datasets
dataset_content = SimpleDataset(args.content_dir, transforms=T.ToTensor())
dataset_style = SimpleDataset(args.style_dir, transforms=T.ToTensor())
sampler_content = InfiniteSamplerWrapper(dataset_content)
sampler_style = InfiniteSamplerWrapper(dataset_style)
dataloader_content_iter = iter(DataLoader(dataset_content,
                      batch_size=args.batch_size,
                      sampler=sampler_content,
                      num_workers=0))
dataloader_style_iter = iter(DataLoader(dataset_style,
                      batch_size=args.batch_size,
                      sampler=sampler_style,
                      num_workers=0))


# Hardware Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
   
# Models
vgg.load_state_dict(torch.load(args.vgg_dir))
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

network = Net(encoder, decoder, transModule, vgg)

# Optimizer
optimizer = torch.optim.Adam([
    {'params': network.encoder.parameters()},
    {'params': network.decoder.parameters()},
    {'params': network.transModule.parameters()},
], lr=args.base_lr)
scheduler = CosineAnnealingWarmUpLR(optimizer, warmup_step=args.epoch//4, max_step=args.epoch, min_lr=0)


# Whether to use parameters from checkpoints
if args.resume_train:
  print('loading checkpoint...')
  checkpoint = torch.load(args.checkpoint_import_path, map_location=device)
  
  network.encoder.load_state_dict(checkpoint['encoder'])
  network.decoder.load_state_dict(checkpoint['decoder'])
  network.transModule.load_state_dict(checkpoint['transModule'])
    
  optimizer.load_state_dict(checkpoint['optimizer'])
  scheduler.load_state_dict(checkpoint['scheduler'])

  for state in optimizer.state.values():
    for k, v in state.items():
      if torch.is_tensor(v):
        state[k] = v.to(device)

  log_c = checkpoint['log_c']
  log_s = checkpoint['log_s']
  log_id1 = checkpoint['log_id1']
  log_id2 = checkpoint['log_id2']
  log_all = checkpoint['log_all']

  epoch_start = checkpoint['epoch']
  loss_count_interval = checkpoint['loss_count_interval']
  print('loading finished')
else:
  log_c, log_s, log_id1, log_id2, log_all = [],[],[],[],[]

log_c_temp, log_s_temp, log_id1_temp, log_id2_temp, log_all_temp = [],[],[],[],[]


# Load the model to device
network.to(device)


# Training
if __name__ == '__main__':
  for i in range(args.epoch):
    i += (epoch_start + 1)
    
    # data samples
    i_c = next(dataloader_content_iter).to(device)
    i_s = next(dataloader_style_iter).to(device)

    # calculate losses
    loss_c, loss_s, loss_id_1, loss_id_2, _ = network(i_c, i_s)
    loss_all = args.content_weight*loss_c + args.style_weight*loss_s + args.id1_weight*loss_id_1 + args.id2_weight*loss_id_2 
    
    log_c_temp.append(loss_c.item())
    log_s_temp.append(loss_s.item())
    log_id1_temp.append(loss_id_1.item())
    log_id2_temp.append(loss_id_2.item())
    log_all_temp.append(loss_all.item())

    # update parameters
    optimizer.zero_grad()
    loss_all.backward()
    optimizer.step()
    scheduler.step()

    # calculate average loss
    if i % loss_count_interval == 0:
      log_c.append(np.mean(np.array(log_c_temp)))
      log_s.append(np.mean(np.array(log_s_temp)))
      log_id1.append(np.mean(np.array(log_id1_temp)))
      log_id2.append(np.mean(np.array(log_id2_temp)))
      log_all.append(np.mean(np.array(log_all_temp)))

      print('Epoch {:d}: '.format(i) + str(log_all[-1]))

      log_c_temp, log_s_temp = [],[]
      log_id1_temp, log_id2_temp = [],[]
      log_all_temp = []

    # save a checkpoint
    if i % args.checkpoint_save_interval == 0:
      save_checkpoint(
        encoder=network.encoder,
        transModule=network.transModule,
        decoder=network.decoder,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=i,
        log_c=log_c,
        log_s=log_s,
        log_id1=log_id1,
        log_id2=log_id2,
        log_all=log_all,
        loss_count_interval=loss_count_interval,
        save_path=os.path.join(args.checkpoint_save_path, 'checkpoint_{}_epoch.pkl'.format(i))
      )