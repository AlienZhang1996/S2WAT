import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os
import zipfile
from PIL import Image
from tqdm import tqdm
from datetime import datetime


####################################### Train Tools #######################################

# save the checkpoint
def save_checkpoint(encoder, transModule, decoder, optimizer, scheduler, epoch,
           log_c, log_s, log_id1, log_id2, log_all, loss_count_interval, save_path):
  checkpoint = {
    'encoder': encoder.state_dict() if not encoder is None else None,
    'transModule': transModule.state_dict() if not transModule is None else None,
    'decoder': decoder.state_dict() if not decoder is None else None,
    'optimizer': optimizer.state_dict() if not optimizer is None else None,
    'scheduler': scheduler.state_dict() if not scheduler is None else None,
    'epoch': epoch if not epoch is None else None,
    'log_c': log_c if not log_c is None else None,
    'log_s': log_s if not log_s is None else None,
    'log_id1': log_id1 if not log_id1 is None else None,
    'log_id2': log_id2 if not log_id2 is None else None,
    'log_all': log_all if not log_all is None else None,
    'loss_count_interval': loss_count_interval if not loss_count_interval is None else None
  }

  torch.save(checkpoint, save_path)


######################################## Test Tools #######################################
def showTorchImage(image):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    mode = transforms.ToPILImage()(image)
    plt.imshow(mode)
    plt.show()
    plt.close()

def zip_dir(zipFile_name, dir_path):
    z = zipfile.ZipFile(zipFile_name, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            z.write(os.path.join(dirpath, filename))
    z.close()


def open_img_to_pt(img_path, transform=transforms.ToTensor()):
    img = Image.open(img_path)
    img_pt = transform(img).unsqueeze(dim=0)
    return img_pt


def content_style_transTo_pt(i_c_path, i_s_path, i_c_size=None):
    """Resize the pics of arbitrary size to the shape of content image
    """
    i_c_pil = Image.open(i_c_path)
    i_s_pil = Image.open(i_s_path)
    
    if not i_c_size is None:
        i_c_tf = transforms.Compose([
            transforms.Resize(i_c_size),
            transforms.ToTensor()
        ])
    else:
        i_c_tf = transforms.Compose([
            transforms.ToTensor()
        ])
    
    i_s_size = min(i_c_pil.size[1], i_c_pil.size[0])
    i_s_tf = transforms.Compose([
        transforms.Resize(i_s_size),
        transforms.ToTensor()
    ])
    
    i_c_pt = i_c_tf(i_c_pil).unsqueeze(dim=0)
    i_s_pt = i_s_tf(i_s_pil).unsqueeze(dim=0)
    
    return i_c_pt, i_s_pt


@torch.no_grad()
def save_sample_imgs(network, samples_path, img_saved_path, device=torch.device('cpu')):
    """Test and save samples imgs (Fixed Size)
       Args:
           network       : Model that tested
           samples_path  : Path where the samples saved
                           Required two sub-dirs named 'Content' and 'Style'
           img_saved_path: Path to save the results
    """
    sample_dict = {
        '1': [1,2,5,6],
        '2': [3,6,9],
        '3': [4,6,9],
        '4': [1,8,9],
        '5': [1,6,8],
        '6': [1,6,7],
        '7': [1,6,9],
        '8': [1,6,8],
        '9': [1,6,7],
    }
    
    print('Image generation starts:')
    for i_c_num in tqdm(sample_dict.keys()):
        output_imgs = torch.tensor([])
        for i_s_num in sample_dict[i_c_num]:     
            i_c = open_img_to_pt(os.path.join(samples_path, f'Content/{i_c_num}.png')).to(device)
            i_s = open_img_to_pt(os.path.join(samples_path, f'Style/{i_s_num}.png')).to(device)
            i_cs = network(i_c, i_s)
            output_img = torch.cat((i_c.cpu(), i_s.cpu(), i_cs.cpu()), dim=0)
            output_imgs = torch.cat((output_imgs, output_img), dim=0)
        output_name = os.path.join(img_saved_path, f'test_{i_c_num}.png')
        save_image(output_imgs, output_name, nrow=3)


@torch.no_grad()
def save_sample_imgs_arbitrarySize(network, samples_path, img_saved_path, device=torch.device('cpu')):
    """Test and save samples imgs (Arbitrary Size)
       Args:
           network       : Model that tested
           samples_path  : Path where the samples saved
                           Required two sub-dirs named 'Content' and 'Style'
           img_saved_path: Path to save the results
    """
    sample_dict = {
        '1': [1,2,3,4,5,6,7,8,9,10,11],
        '2': [1,2,3,4,5,6,7,8,9,10,11],
    }
    
    print('Image generation starts:')
    for i_c_num in tqdm(sample_dict.keys()):
        i_cs = torch.tensor([])
        output_imgs = torch.tensor([])
        i_c_path = os.path.join(samples_path, f'Content/{i_c_num}.jpg')
        for i_s_num in sample_dict[i_c_num]: 
            i_s_path = os.path.join(samples_path, f'Style/{i_s_num}.jpg')
            i_c, i_s = content_style_transTo_pt(i_c_path, i_s_path)
            i_cs = network(i_c.to(device), i_s.to(device), arbitrary_input=True)
            i_s = transforms.CenterCrop((i_c.shape[2], i_c.shape[3]))(i_s)
            i_cs = transforms.CenterCrop((i_c.shape[2], i_c.shape[3]))(i_cs)
            output_img = torch.cat((i_c.cpu(), i_s.cpu(), i_cs.cpu()), dim=0)
            output_imgs = torch.cat((output_imgs, output_img), dim=0)
        output_name = os.path.join(img_saved_path, f'test_{i_c_num}.jpg')
        save_image(output_imgs, output_name, nrow=3)


@torch.no_grad()
def save_transferred_imgs(network, samples_path, img_saved_path, device=torch.device('cpu')):
  print('Image generation starts:')

  i_c_names = os.listdir(os.path.join(samples_path, 'Content'))
  i_s_names = os.listdir(os.path.join(samples_path, 'Style'))
  for i_c_name in tqdm(i_c_names):
    for i_s_name in tqdm(i_s_names):
      i_c_path = os.path.join(samples_path, 'Content', i_c_name)
      i_s_path = os.path.join(samples_path, 'Style', i_s_name)
      i_c, i_s = content_style_transTo_pt(i_c_path, i_s_path)
      i_cs = network(i_c.to(device), i_s.to(device), arbitrary_input=True)

      stem_c, suffix_c = os.path.splitext(i_c_name)
      stem_s, suffix_s = os.path.splitext(i_s_name)
      output_name = os.path.join(img_saved_path, f'{stem_c}_+_{stem_s}.{suffix_c}')
      save_image(i_cs, output_name)


@torch.no_grad()
def save_content_leak_imgs(network, samples_path, img_saved_path, rounds=20, device=torch.device('cpu')):
  print('Image generation starts:')

  i_c_names = os.listdir(os.path.join(samples_path, 'Content'))
  i_s_names = os.listdir(os.path.join(samples_path, 'Style'))
  for i_c_name in tqdm(i_c_names):
    for i_s_name in tqdm(i_s_names):
      i_c_path = os.path.join(samples_path, 'Content', i_c_name)
      i_s_path = os.path.join(samples_path, 'Style', i_s_name)
      i_c, i_s = content_style_transTo_pt(i_c_path, i_s_path)

      i_c = i_c.to(device)
      i_s = i_s.to(device)
      i_cs = i_c
      for i in range(rounds):
        i_cs = network(i_cs, i_s, arbitrary_input=True)

      stem_c, suffix_c = os.path.splitext(i_c_name)
      stem_s, suffix_s = os.path.splitext(i_s_name)
      output_name = os.path.join(img_saved_path, f'{stem_c}_+_{stem_s}.{suffix_c}')
      save_image(i_cs, output_name)


@torch.no_grad()
def caculate_avg_generate_time(network, i_c_path, i_s_path, round=1, device=torch.device('cpu')):
  i_c = open_img_to_pt(i_c_path)
  i_s = open_img_to_pt(i_s_path)
  i_c = i_c.to(device)
  i_s = i_s.to(device)
  
  time_start = datetime.now()
  for i in range(round):
    i_cs = network(i_c, i_s, arbitrary_input=True)
  time_end = datetime.now()

  avg_generate_time = ((time_end-time_start).seconds + (time_end-time_start).microseconds/1000000) / round
  return avg_generate_time


@torch.no_grad()
def caculate_avg_generate_time_multiple(network, samples_path, device=torch.device('cpu')):
  i_c_names = os.listdir(os.path.join(samples_path, 'Content'))
  i_s_names = os.listdir(os.path.join(samples_path, 'Style'))

  nums = len(i_c_names) * len(i_s_names)

  time_start = datetime.now()
  for i_c_name in i_c_names:
    for i_s_name in i_s_names:
      i_c_path = os.path.join(samples_path, 'Content', i_c_name)
      i_s_path = os.path.join(samples_path, 'Style', i_s_name)
      i_c, i_s = content_style_transTo_pt(i_c_path, i_s_path)

      i_c = i_c.to(device)
      i_s = i_s.to(device)
      i_cs = network(i_c, i_s, arbitrary_input=True)
  time_end = datetime.now()
      
  avg_generate_time = ((time_end-time_start).seconds + (time_end-time_start).microseconds/1000000) / nums
  return avg_generate_time


class Sample_Test_Net(nn.Module):
    def __init__(self, encoder, decoder, transModule, patch_size=8):
        super(Sample_Test_Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transModule = transModule
        self.patch_size = patch_size

    def forward(self, i_c, i_s, arbitrary_input=False):
        _, _, H, W = i_c.size()
        self.decoder.img_H = H
        self.decoder.img_W = W
        f_c = self.encoder(i_c, arbitrary_input)
        f_s = self.encoder(i_s, arbitrary_input)
        f_c, f_c_reso = f_c[0], f_c[2]
        f_s, f_s_reso = f_s[0], f_s[2]
        f_cs = self.transModule(f_c, f_s)
        i_cs = self.decoder(f_cs, f_c_reso)
        return i_cs