from ddpm import DDPM
from vae import Decoder
import sys
import os
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Sample LDM")

parser.add_argument('-dp', '--ddpmpath', default='./ddpm.pt')
parser.add_argument('-decp', '--decpath', default='./vae_decoder.pt')
parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-s', '--size', default=256, type=int)
parser.add_argument('-n', '--numimages', default=1, type=int)
parser.add_argument('-t', '--timesteps', default=30, type=int)
parser.add_argument('--seed', default=0, type=int)

args = parser.parse_args()

ddpm_path = args.ddpmpath
vae_decoder_path = args.decpath
image_size = args.size
latent_space_downscale_ratio = 8
result_dir = "./ddpm_outputs/"
num_images = args.numimages
use_autocast = args.fp16

device_name = args.device

print(f"selected device: {device_name}")
if device_name == 'cuda':
    if not torch.cuda.is_available():
        print("Error: cuda is not available in this environment.")
        exit()

if device_name == 'mps':
    if not torch.backends.mps.is_built():
        print("Error: mps is not available in this environment.")
        exit()

device = torch.device(device_name)
ddpm = DDPM()
decoder = Decoder()

if os.path.exists(ddpm_path):
    ddpm.load_state_dict(torch.load(ddpm_path, map_location='cpu'))
    print("DDPM Model Loaded.")

if os.path.exists(vae_decoder_path):
    decoder.load_state_dict(torch.load(vae_decoder_path, map_location='cpu'))
    print("VAE Decoder Loaded.")


ddpm = ddpm.to(device)
decoder = decoder.to(device)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# Convert to latent size
image_size = image_size // latent_space_downscale_ratio
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

for i in range(num_images):
    img = ddpm.sample((1, 8, image_size, image_size), seed=None, num_steps=args.timesteps, use_autocast = use_autocast)
    with torch.no_grad():
        img = decoder(img)
    img = torch.clamp(img, -1, 1)
    path = os.path.join(result_dir, f"{i}.jpg")
    img = Image.fromarray((img[0].cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
    img.save(path)
