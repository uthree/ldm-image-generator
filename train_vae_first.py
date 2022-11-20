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
discriminator_path = "./vae_discriminator.pt"
result_dir = "./vae_result/"
batch_size = 2
num_epoch = 3000
learning_rate = 1e-4
image_size = 512
crop_size = (192, 192)
num_crop_per_batch = 2
max_dataset_size = 20000
weight_kl = 1.0
weight_adv = 2.0
weight_recon = 3.0
weight_feat = 5.0
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
    print("Discriminator Model Loaded.")

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

vae = VAE(encoder, decoder)
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
        
        for i in range(num_crop_per_batch):
            # Train G.
            optimizer_vae.zero_grad()
            image = image.to(device)
            image = crop(image)
        
            with torch.cuda.amp.autocast(enabled=use_autocast):
                recon_loss, kl_loss, y = vae.calclate_loss(image)
                logit, feat_loss = discriminator.calclate_logit_and_feature_matching(y, image)
                adv_loss = F.relu(-logit).mean()
                loss = recon_loss * weight_recon + kl_loss * weight_kl + adv_loss * weight_adv + feat_loss * weight_feat
            scaler.scale(loss).backward()
            scaler.step(optimizer_vae)
            fake = y.detach()

            # Train D.
            optimizer_d.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_autocast):
                fake_logit = discriminator.calclate_logit(fake)
                real_logit = discriminator.calclate_logit(image)
                loss_d = F.relu(0.5-real_logit).mean() + F.relu(0.5+fake_logit).mean()
            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)

            scaler.update()
            bar.set_description(desc=f"Recon: {recon_loss.item():.4f}, KL {kl_loss.item():.4f}, Adv: {adv_loss.item():.4f}, Feat: {feat_loss.item():.4f}, Disc: {loss_d.item():.4f}")
            bar.update(0)
        bar.update(N)
        if batch % 100 == 0:
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)
            torch.save(discriminator.state_dict(), discriminator_path)
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


