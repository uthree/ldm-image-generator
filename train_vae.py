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
batch_size = 8
num_epoch = 100
lr = 2e-5
image_size = 256
train_adv = True
use_autocast = True

ds = ImageDataset(sys.argv[1:], max_len=10000, size=image_size)
vae = VAE()

if os.path.exists(vae_path):
    vae.load_state_dict(torch.load(vae_path))
    print("VAE Model Loaded.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae.to(device)
if train_adv:
    disc = Discriminator()
    if os.path.exists(discriminator_path):
        disc.load_state_dict(torch.load(discriminator_path))
        disc.to(device)
        print("Discriminator Model Loaded.")
    vae.train_decoder(ds, batch_size=batch_size, num_epoch=num_epoch, lr=lr, discriminator=disc)
else:
    vae.train_encdec(ds, batch_size=batch_size, num_epoch=num_epoch, lr=lr)
