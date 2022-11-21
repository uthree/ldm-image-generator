from dataset import LatentImageDataset
from vae import Encoder
from ddpm import DDPM
import sys
import os
from tqdm import tqdm
import torch
import torch.optim as optim

ddpm_path = "./ddpm.pt"
vae_encoder_path = "./vae_encoder.pt"
batch_size = 8
num_epoch = 3000
learning_rate = 1e-4
image_size = 512
max_dataset_size = 1000
use_autocast = True

ddpm = DDPM()
encoder = Encoder()

if os.path.exists(ddpm_path):
    ddpm.load_state_dict(torch.load(ddpm_path))
    print("DDPM Model Loaded.")

if os.path.exists(vae_encoder_path):
    encoder.load_state_dict(torch.load(vae_encoder_path))
    print("VAE Encoder Loaded.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

ds = LatentImageDataset(sys.argv[1:], max_len=max_dataset_size, size=image_size, encoder=encoder, device=device)
del encoder

ddpm.to(device)
optimizer = optim.RAdam(ddpm.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_autocast)

dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
for epoch in range(num_epoch):
    bar = tqdm(total=len(ds))
    print(f"Epoch #{epoch}")
    for batch, image in enumerate(dl):
        N = image.shape[0]
        optimizer.zero_grad()
        image = image.to(device)
        
        with torch.cuda.amp.autocast(enabled=use_autocast):
            ddpm_loss = ddpm.calculate_loss(image)
            loss = ddpm_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
    
        scaler.update()
        bar.set_description(desc=f"loss: {ddpm_loss.item():.4f}")
        bar.update(N)
        if batch % 300 == 0:
            torch.save(ddpm.state_dict(), ddpm_path)
