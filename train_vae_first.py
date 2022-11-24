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

encoder_path = "./vae_encoder.pt"
decoder_path = "./vae_decoder.pt"
discriminator_path = "./discriminator.pt"
result_dir = "./vae_result/"
batch_size = 4
num_epoch = 3000
learning_rate = 1e-4
image_size = 512
crop_size = (192, 192)
num_crop_per_batch = 1
max_dataset_size = 10000
weight_kl = 1.0
weight_recon = 1.0
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
    discriminator.load_state_disct(torch.load(discriminator_path))

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

vae = VAE(encoder, decoder)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
vae.to(device)
discriminator.to(device)

optimizer_vae = optim.RAdam(vae.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_autocast)
optimizer_d = optim.RAdam(discriminator.parameters(), lr=learning_rate)
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
                recon_loss, kl_loss, y = vae.calclate_loss(image)
                adv_loss = (-discriminator.calclate_logit(y)).mean()
                loss = recon_loss * weight_recon + kl_loss * weight_kl + adv_loss * weight_adv

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
            bar.set_description(desc=f"Recon: {recon_loss.item():.4f}, KL {kl_loss.item():.4f}, Adv.: {adv_loss.item():.4f}, Disc.: {d_loss.item():.4f}")
            bar.update(0)
        bar.update(N)
        if batch % 100 == 0:
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)
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


