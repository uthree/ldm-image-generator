from ddpm import DDPM
from vae import Decoder
import sys
import os
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np

ddpm_path = "./ddpm.pt"
vae_decoder_path = "vae_decoder.pt"
image_size = 512
latent_space_downscale_ratio = 8
result_dir = "./ddpm_outputs/"
num_images = 30
use_cpu = True

ddpm = DDPM()
decoder = Decoder()

if os.path.exists(ddpm_path):
    ddpm.load_state_dict(torch.load(ddpm_path))
    print("DDPM Model Loaded.")

if os.path.exists(vae_decoder_path):
    decoder.load_state_dict(torch.load(vae_decoder_path))
    print("VAE Decoder Loaded.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if use_cpu:
    device = torch.device('cpu')

print(f"device: {device}")
ddpm.to(device)
decoder.to(device)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# Convert to latent size
image_size = image_size // latent_space_downscale_ratio

for i in range(num_images):
    img = ddpm.sample((1, 4, image_size, image_size), seed=i, num_steps=30)
    with torch.no_grad():
        img = decoder(img)
    img = torch.clamp(img, -1, 1)
    path = os.path.join(result_dir, f"{i}.jpg")
    img = Image.fromarray((img[0].cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
    img.save(path)
