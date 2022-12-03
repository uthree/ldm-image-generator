from dataset import ImageDataset
from vae import VAE, Encoder, Decoder, Discriminator, VectorQuantizer
import torch.nn.functional as F
import torchvision.transforms as T
import sys
import os
from tqdm import tqdm
import torch
from transformers import Adafactor
from PIL import Image
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Train VAE")

parser.add_argument('dataset_path')
parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                    help="Device setting. Set this option to cuda if you need to use ROCm.")
parser.add_argument('-e', '--epoch', default=1, type=int)
parser.add_argument('-b', '--batch', default=1, type=int)
parser.add_argument('-r', '--result', default='./results')
parser.add_argument('-ep', '--encpath', default='./vae_encoder.pt')
parser.add_argument('-dp', '--decpath', default='./vae_decoder.pt')
parser.add_argument('-qp', '--quantizerpath', default='vae_quantizer.pt')
parser.add_argument('-discp', '--discpath', default='./discriminator.pt')
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('-s', '--size', default=512, type=int)
parser.add_argument('-m', '--maxdata', default=-1, type=int, help="max dataset size")
parser.add_argument('--recon', default=10, type=float)

args = parser.parse_args()

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

encoder_path = args.encpath
decoder_path = args.decpath
quantizer_path = args.quantizerpath
discriminator_path = args.discpath
result_dir = args.result

batch_size = args.batch
num_epoch = args.epoch
image_size = args.size
crop_size = (192, 192)
num_crop_per_batch = 1
max_dataset_size = args.maxdata
weight_reg = 1.0
weight_recon = args.recon
weight_adv = 0.1
use_autocast = args.fp16

ds = ImageDataset([args.dataset_path], max_len=max_dataset_size, size=image_size)
encoder = Encoder()
decoder = Decoder()
quantizer = VectorQuantizer()
discriminator = Discriminator()
crop = T.RandomCrop(crop_size)

if os.path.exists(encoder_path):
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    print("Encoder Model Loaded.")

if os.path.exists(decoder_path):
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    print("Decoder Model Loaded.")

if os.path.exists(discriminator_path):
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    print("Discriminator Model Loaded.")

if os.path.exists(quantizer_path):
    quantizer.load_state_dict(torch.load(quantizer_path, map_location=device))
    print("Quantizer Model Loaded.")
    
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

vae = VAE(encoder, decoder, quantizer)
vae.to(device)
discriminator.to(device)

optimizer_vae = Adafactor(vae.parameters())
scaler = torch.cuda.amp.GradScaler(enabled=use_autocast)
optimizer_d = Adafactor(discriminator.parameters())
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
for epoch in range(num_epoch):
    bar = tqdm(total=len(ds))
    print(f"Epoch #{epoch}")
    for batch, image in enumerate(dl):
        N = image.shape[0]
        
        for i in range(num_crop_per_batch):
            # Train VAE.
            optimizer_vae.zero_grad()
            image = image.to(device)
            image = crop(image)
        
            with torch.cuda.amp.autocast(enabled=use_autocast):
                recon_loss, reg_loss, y = vae.calclate_loss(image)
                adv_loss = F.relu(-discriminator.calclate_logit(y)).mean()
                loss = recon_loss * weight_recon + reg_loss * weight_reg + adv_loss * weight_adv

            scaler.scale(loss).backward()
            scaler.step(optimizer_vae)
            # Train D.
            optimizer_d.zero_grad()
            y = y.detach()
            with torch.cuda.amp.autocast(enabled=use_autocast):
                logit_fake, logit_real = discriminator.calclate_logit(y), discriminator.calclate_logit(image)
                logit_fake = F.relu(1+logit_fake).mean()
                logit_real = F.relu(1-logit_real).mean()
                d_loss = logit_fake + logit_real
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)

            scaler.update()
            bar.set_description(desc=f"Recon: {recon_loss.item():.4f}, Reg {reg_loss.item():.4f}, Adv.: {adv_loss.item():.4f}, Disc.: {d_loss.item():.4f}")
            bar.update(0)
        bar.update(N)
        if batch % 100 == 0:
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)
            torch.save(discriminator.state_dict(), discriminator_path)
            torch.save(quantizer.state_dict(), quantizer_path)
            img = torch.clamp(y[0].detach(), -1, 1)

            # Save Reconstructedd image
            path = os.path.join(result_dir, f"{batch}_reconstructed.jpg")
            img = Image.fromarray((img.cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
            img.save(path)
            
            # Save Input image
            img = torch.clamp(image[0].detach(), -1, 1)
            path = os.path.join(result_dir, f"{batch}_input.jpg")
            img = Image.fromarray((img.cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
            img.save(path)


