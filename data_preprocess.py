# -*- coding: utf-8 -*-
"""
@Function: Preprocess the datasets
    1.Preprocess Steps: Resize images to 512 on the shorter side and randomly crop to 224x224
    2.Excluded Objects：Corrupted or oversize images (names will be printed in command)
@Attention: 
    1. source_dir and target_dir should exists,
    2. source_dir needs a subfolder of any name to load images, such as "./source/images"
    3. images to be processed should be in source_dir
"""

import argparse
import os
import torch
import torchvision
from torchvision import transforms as T
from pathlib import Path

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--source_dir', type=str, required=True,
                    help='Directory to the images to be processed')
parser.add_argument('--target_dir', type=str, default='./target',
                    help='Directory to save the processed images')

args = parser.parse_args()

# Print args
print('Running args: ')
for k, v in sorted(vars(args).items()):
    print(k, '=', v)
print()

Path(args.target_dir).mkdir(exist_ok=True, parents=True)

source_dir = args.source_dir
target_dir = args.target_dir


# Preprocess Steps
# 1. Resize images to 512 on the shorter side
# 2. Randomly crop to 224x224
transform = T.Compose([
    T.Resize(512),
    T.RandomCrop((224,224)),
    T.ToTensor()
])

# Dataset
dataset = torchvision.datasets.ImageFolder(source_dir, transform=transform)

# Sampler
seq_sampler = torch.utils.data.SequentialSampler(dataset)

if __name__ == "__main__":

    print("Images unprocessed: ")
    for index in seq_sampler:
        try:
            image,label = dataset[index]
            imageName = os.path.join(target_dir, os.path.basename(dataset.imgs[index][0]).split(".")[0] + ".png")
            torchvision.utils.save_image(image, imageName, nrow=1, padding=0)
        except Exception:
            # Print the names of corrupted or oversize images
            print(dataset.imgs[index][0])