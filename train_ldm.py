from dataset import LatentImageDataset
from vae import Encoder
from ddpm import DDPM
import sys
import os
from tqdm import tqdm
import torch
from lion_pytorch import Lion
import argparse

parser = argparse.ArgumentParser(description="Train Latent Diffusion Model")

parser.add_argument('dataset_path')
parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-e', '--epoch', default=1, type=int)
parser.add_argument('-b', '--batch', default=1, type=int)
parser.add_argument('-mp', '--modelpath', default='./ddpm.pt')
parser.add_argument('-ep', '--encpath', default='./vae_encoder.pt')
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-s', '--size', default=512, type=int)
parser.add_argument('-m', '--maxdata', default=-1, type=int, help="max dataset size")
parser.add_argument('-lr', '--learningrate', default=1e-4, type=float)
parser.add_argument('-bm', '--batch_multiply', default=1, type=int)

args = parser.parse_args()
device_name = args.device

ddpm_path = args.modelpath
vae_encoder_path = args.encpath
batch_size = args.batch
num_epoch = args.epoch
learning_rate = args.learningrate
image_size = args.size
max_dataset_size = args.maxdata
use_autocast = args.fp16
bm = args.batch_multiply

ddpm = DDPM()
encoder = Encoder()

if os.path.exists(ddpm_path):
    ddpm.load_state_dict(torch.load(ddpm_path))
    print("DDPM Model Loaded.")

if os.path.exists(vae_encoder_path):
    encoder.load_state_dict(torch.load(vae_encoder_path))
    print("VAE Encoder Loaded.")

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

ds = LatentImageDataset(sys.argv[1:], max_len=max_dataset_size, size=image_size, encoder=encoder, device=device)
del encoder

ddpm.to(device)
optimizer = Lion(ddpm.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_autocast)

dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
for epoch in range(num_epoch):
    bar = tqdm(total=len(ds))
    print(f"Epoch #{epoch}")
    for batch, image in enumerate(dl):
        N = image.shape[0]
        if batch % bm == 0:
            optimizer.zero_grad()
        image = image.to(device)
        
        with torch.cuda.amp.autocast(enabled=use_autocast):
            ddpm_loss = ddpm.calculate_loss(image)
            loss = ddpm_loss
        scaler.scale(loss).backward()
        if batch % bm == 0:
            scaler.step(optimizer)
            scaler.update()

        bar.set_description(desc=f"loss: {ddpm_loss.item():.4f}")
        bar.update(N)
        if batch % 300 == 0:
            tqdm.write('Model is saved!')
            torch.save(ddpm.state_dict(), ddpm_path)
