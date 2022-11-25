from dataset import ImageDataset
from vae import VAE, Encoder, Decoder, Discriminator
import torch.nn.functional as F
import torchvision.transforms as T
import sys
import os
from tqdm import tqdm
import torch
import torch.optim as optim
from PIL import Image
import numpy as np
from transformers import Adafactor

encoder_path = "./vae_encoder.pt"
decoder_path = "./vae_decoder.pt"
discriminator_path = "./vae_discriminator.pt"
result_dir = "./vae_result/"
batch_size = 8
num_epoch = 3000
learning_rate = 1e-4
image_size = 256
crop_size = (192, 192)
num_crop_per_batch = 1
max_dataset_size = 10000
weight_recon = 5.0
weight_feat = 1.0
weight_adv = 1.0
use_autocast = True

ds = ImageDataset(sys.argv[1:], max_len=max_dataset_size, size=image_size)
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()
crop = T.RandomCrop(crop_size)

if os.path.exists(encoder_path):
    encoder.load_state_dict(torch.load(encoder_path))
    print("Encoder Model Loaded.")

if os.path.exists(decoder_path):
    decoder.load_state_dict(torch.load(decoder_path))
    print("Decoder Model Loaded.")

if os.path.exists(discriminator_path):
    discriminator.load_state_dict(torch.load(discriminator_path))
    print("Discriminator Model Loaded")

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

vae = VAE(encoder, decoder)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
vae.to(device)
discriminator.to(device)
optimizer_vae = Adafactor(vae.parameters(), lr=learning_rate)
optimizer_d = Adafactor(discriminator.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_autocast)
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

for epoch in range(num_epoch):
    bar = tqdm(total=len(ds))
    print(f"Epoch #{epoch}")
    for batch, image in enumerate(dl):
        N = image.shape[0]
        
        for i in range(num_crop_per_batch):
            # Train G.
            optimizer_vae.zero_grad()
            image = image.to(device)
            image = crop(image)
        
            with torch.cuda.amp.autocast(enabled=use_autocast):
                z = vae.encode(image)
                fake = vae.decoder.forward(z)
                loss_g_recon = (fake-image).abs().mean()
                logit, loss_g_feat = discriminator.calclate_logit_and_feature_matching(fake, image)
                loss_g_adv = (-logit).mean()
                loss_g = loss_g_recon * weight_recon + loss_g_feat * weight_feat + loss_g_adv * weight_adv
            scaler.scale(loss_g).backward()
            torch.nn.utils.clip_grad_norm_(vae.decoder.parameters(), 1, norm_type=2.0)
            scaler.step(optimizer_vae)

            # Train D.
            optimizer_d.zero_grad()
            fake = fake.detach()
            with torch.cuda.amp.autocast(enabled=use_autocast):
                logit_fake = discriminator.calclate_logit(fake)
                logit_real = discriminator.calclate_logit(image)
                loss_d = F.relu(1-logit_real).mean() + F.relu(1+logit_fake).mean()
            scaler.scale(loss_d).backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1, norm_type=2.0)
            scaler.step(optimizer_d)

            scaler.update()
            bar.set_description(desc=f"Recon: {loss_g_recon.item():.4f}, Feat: {loss_g_feat.item():.4f}, Adv: {loss_g_adv.item():.4f}, Disc.: {loss_d.item():.4f}")
            bar.update(0)
        bar.update(N)
        if batch % 100 == 0:
            torch.save(discriminator.state_dict(), discriminator_path)
            torch.save(decoder.state_dict(), decoder_path)
            img = torch.clamp(fake[0].detach(), -1, 1)

            # Save Reconstructedd image
            path = os.path.join(result_dir, f"{batch}_reconstructed.jpg")
            img = Image.fromarray((img.cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
            img.save(path)
            
            # Save Input image
            img = torch.clamp(image[0].detach(), -1, 1)
            path = os.path.join(result_dir, f"{batch}_input.jpg")
            img = Image.fromarray((img.cpu().numpy() * 127.5 + 127.5).astype(np.uint8).transpose(1,2,0), mode='RGB')
            img.save(path)

