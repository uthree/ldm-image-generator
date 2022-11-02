from dataset import ImageDataset
from vae import VAE, Discriminator
import sys
import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

vae_path = "./vae.pt"
discriminator_path = "./discriminator.pt"
batch_size = 1
num_epoch = 30
learning_rate = 1e-4
image_size = 256
use_autocast = True
weight_recon = 10
weight_kl = 1

ds = ImageDataset(sys.argv[1:], max_len=1000, size=image_size)
vae = VAE()
discriminator = Discriminator()

if os.path.exists(vae_path):
    vae.load_state_dict(torch.load(vae_path))
    print("VAE Model Loaded.")

if os.path.exists(discriminator_path):
    discriminator.load_state_dict(torch.load(discriminator_path))
    print("Discriminator Model Loaded.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
vae.to(device)
discriminator.to(device)
optimizer_vae = optim.RAdam(vae.parameters(), lr=learning_rate)
optimizer_d = optim.RAdam(discriminator.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_autocast)

dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
for epoch in range(num_epoch):
    bar = tqdm(total=len(ds))
    print(f"Epoch #{epoch}")
    for batch, image in enumerate(dl):
        N = image.shape[0]
        optimizer_vae.zero_grad()
        image = image.to(device)

        with torch.cuda.amp.autocast(enabled=use_autocast):
            loss_recon, loss_kl, y = vae.caluclate_loss(image)
            loss_adv = F.relu(-discriminator(y)).mean()
        loss = loss_recon * weight_recon + loss_kl * weight_kl + loss_adv
        nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0, norm_type=2.0)
        loss = scaler.scale(loss)
        loss.backward()
        scaler.step(optimizer_vae)

        y = y.detach()
        optimizer_d.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_autocast):
            loss_fake = F.relu(0.5 + discriminator(y)).mean()
            loss_real = F.relu(0.5 - discriminator(image)).mean()
        loss_d = loss_fake + loss_real
        nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0, norm_type=2.0)
        loss = scaler.scale(loss_d)
        loss.backward()
        scaler.step(optimizer_d)
    
        scaler.update()
        bar.set_description(desc=f"Recon: {loss_recon.item():.4f}, KL: {loss_kl.item():.4f}, Adv: {loss_adv.item():.4f}, Disc. {loss_d.item():.4f}")
        bar.update(N)
        if batch % 1000 == 0:
            torch.save(vae.state_dict(), vae_path)
            torch.save(discriminator.state_dict(), discriminator_path)
