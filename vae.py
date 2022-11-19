import torch
import torch.nn as nn
import torch.nn.Functional as F
from attention import WindowAttention

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def caluclate_loss(self, x):
        mean, logvar = self.encoder(x)
        var = torch.exp(logvar)
        loss_kl = -1 - logvar + var + mean ** 2
        z = torch.randn_like(mean) * var + mean
        y = self.decoder(z)
        loss_recon = (x-y).abs()
        return loss_recon, loss_kl
    
    @torch.no_grad()
    def encode(self, x, sigma=0):
        mean, logvar = self.encoder(x)
        var = torch.exp(logvar)
        z = torch.randn_like(mean) * var * sigma + mean
        return z
    
    @ torch.no_grad()
    def decode(self, z):
        return self.decoder(z)

def EncoderBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.ReLU()
        self.c2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        return self.c2(self.act(self.c1(x))) + x

def EncoderStack(nn.Module):
    def __init__(self, channels, num_layers=2):
        super().__init__()
        self.seq = nn.Sequential(*[EncoderBlock(channels) for _ in range(num_layers)])

    def forward(self, x):
        return self.seq(x)
