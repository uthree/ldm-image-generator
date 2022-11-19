from dataset import ImageDataset
from vae import VAE, Encoder, Decoder
import sys
import os
from tqdm import tqdm
import torch
import torch.optim as optim

encoder_path = "./vae_encoder.pt"
decoder_path = "./vae_decoder.pt"
batch_size = 1
num_epoch = 3000
learning_rate = 1e-4
image_size = 512
max_dataset_size = 10000
weight_kl = 0.2
use_autocast = True

ds = ImageDataset(sys.argv[1:], max_len=max_dataset_size, size=image_size)
encoder = Encoder()
decoder = Decoder()

if os.path.exists(encoder_path):
    encoder.load_state_dict(torch.load(encoder_path))
    print("Encoder Model Loaded.")

if os.path.exists(decoder_path):
    decoder.load_state_dict(torch.load(decoder_path))
    print("Decoder Model Loaded.")

vae = VAE(encoder, decoder)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
vae.to(device)
optimizer = optim.RAdam(vae.parameters(), lr=learning_rate)
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
            recon_loss, kl_loss = vae.calclate_loss(image)
            loss = recon_loss + kl_loss * weight_kl
        scaler.scale(loss).backward()
        scaler.step(optimizer)
    
        scaler.update()
        bar.set_description(desc=f"Recon: {recon_loss.item():.4f}, KL {kl_loss.item():.4f}")
        bar.update(N)
        if batch % 300 == 0:
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)
