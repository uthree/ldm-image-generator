from dataset import ImageDataset
from ddpm import DDPM
import sys
import os
from tqdm import tqdm
import torch
import torch.optim as optim

ddpm_path = "./ddpm.pt"
batch_size = 16
num_epoch = 3000
learning_rate = 1e-4
image_size = 32
use_autocast = True

ds = ImageDataset(sys.argv[1:], max_len=5000, size=image_size)
ddpm = DDPM()

if os.path.exists(ddpm_path):
    ddpm.load_state_dict(torch.load(ddpm_path))
    print("DDPM Model Loaded.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
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
            ddpm_loss = ddpm.caluclate_loss(image)
            loss = ddpm_loss
            loss = scaler.scale(loss)
            loss.backward()
        scaler.step(optimizer)
    
        scaler.update()
        bar.set_description(desc=f"loss: {ddpm_loss.item():.4f}")
        bar.update(N)
        if batch % 1000 == 0:
            torch.save(ddpm.state_dict(), ddpm_path)
