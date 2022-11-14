from dataset import ImageDataset
from ddpm import DDPM
from vae import VAE
from ldm import LDM
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
vae_path = "./vae.pt"
image_size = 32
result_dir = "./ldm_results/"
num_images = 100
use_cpu = False

ddpm = DDPM()
if os.path.exists(ddpm_path):
    ddpm.load_state_dict(torch.load(ddpm_path))
    print("DDPM Model Loaded.")

vae = VAE()
if os.path.exists(vae_path):
    vae.load_state_dict(torch.load(vae_path))
    print("VAE Model Loaded.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if use_cpu:
    device = torch.device('cpu')

print(f"device: {device}")
ddpm.to(device)
vae.to(device)

ldm = LDM(ddpm=ddpm, vae=vae)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

for i in range(num_images):
    img = ldm.sample((1, 4, image_size, image_size), seed=None)
    img = torch.clamp(img, -1, 1)
    path = os.path.join(result_dir, f"{i}.jpg")
    img = Image.fromarray((img[0].cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
    img.save(path)
