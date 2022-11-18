from dataset import ImageDataset
from ddpm import DDPM
import sys
import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np

ddpm_path = "./ddpm.pt"
image_size = 32
result_dir = "./results/"
num_images = 10
use_cpu = False

ddpm = DDPM()

if os.path.exists(ddpm_path):
    ddpm.load_state_dict(torch.load(ddpm_path))
    print("DDPM Model Loaded.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if use_cpu:
    device = torch.device('cpu')

print(f"device: {device}")
ddpm.to(device)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

for i in range(num_images):
    img = ddpm.sample_implicitly((1, 3, image_size, image_size), seed=i)
    img = torch.clamp(img, -1, 1)
    path = os.path.join(result_dir, f"{i}.jpg")
    img = Image.fromarray((img[0].cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
    img.save(path)
