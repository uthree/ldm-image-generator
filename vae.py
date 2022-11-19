import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ChannelNorm, ReGLU

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def calclate_loss(self, x):
        mean, logvar = self.encoder(x)
        var = torch.exp(logvar)
        loss_kl = (-1 - logvar + var + mean ** 2).mean()
        z = torch.randn_like(mean) * torch.sqrt(var) + mean
        y = self.decoder(z)
        loss_recon = (x-y).abs().mean()
        return loss_recon, loss_kl
    
    @torch.no_grad()
    def encode(self, x, sigma=0):
        mean, logvar = self.encoder(x)
        var = torch.exp(logvar)
        z = torch.randn_like(mean) * torch.sqrt(var) * sigma + mean
        return z
    
    @ torch.no_grad()
    def decode(self, z):
        return self.decoder(z)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, 7, 1, 3, groups=channels)
        self.norm = ChannelNorm(channels)
        self.ff = ReGLU(channels, ffn_mul=2)

    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.ff(x)
        return x + res

class ResStack(nn.Module):
    def __init__(self, channels, num_layers=2):
        super().__init__()
        self.seq = nn.Sequential(*[ResBlock(channels) for _ in range(num_layers)])

    def forward(self, x):
        return self.seq(x)

class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_channels=16, channels=[32, 64, 128, 256], stages=[2, 2, 2, 2]):
        super().__init__()
        self.input_layer = nn.Conv2d(input_channels, channels[0], 1, 1, 0)
        self.output_layer = nn.Conv2d(channels[-1], latent_channels*2, 1, 1, 0)
        self.stages = nn.ModuleList([ResStack(c, l) for c, l in zip(channels, stages)])
        self.downsamples = nn.ModuleList([])
        for i, c in enumerate(channels):
            if i == len(self.stages)-1:
                self.downsamples.append(nn.Identity())
            else:
                self.downsamples.append(nn.Sequential(
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(c, channels[i+1], 1, 1, 0)))

    def forward(self, x):
        x = self.input_layer(x)
        for a, b in zip(self.stages, self.downsamples):
            x = a(x)
            x = b(x)
        mean, logvar = torch.chunk(self.output_layer(x), 2, dim=1)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, output_channels=3, latent_channels=16, channels=[512, 256, 128, 64], stages=[4, 3, 2, 2], window_size=4):
        super().__init__()
        self.input_layer = nn.Conv2d(latent_channels, channels[0], 1, 1, 0)
        self.output_layer = nn.Conv2d(channels[-1], output_channels, 1, 1, 0)
        self.stages = nn.ModuleList([ResStack(c, l) for c, l in zip(channels, stages)])
        self.upsamples = nn.ModuleList([])
        for i, c in enumerate(channels):
            if i == len(self.stages)-1:
                self.upsamples.append(nn.Identity())
            else:
                self.upsamples.append(nn.ConvTranspose2d(c, channels[i+1], 2, 2, 0))

    def forward(self, x):
        x = self.input_layer(x)
        for a, b in zip(self.stages, self.upsamples):
            x = a(x)
            x = b(x)
        x = self.output_layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, channels=[32, 64, 128, 256], stages=[2, 2, 2, 2], stem_size=2):
        super().__init__()
        self.input_layer = nn.Conv2d(input_channels, channels[0], stem_size, stem_size, 0)
        self.output_layer = nn.Conv2d(channels[-1], 1, 1, 1, 0)
        self.stages = nn.ModuleList([ResStack(c, l) for c, l in zip(channels, stages)])
        self.downsamples = nn.ModuleList([])
        for i, c in enumerate(channels):
            if i == len(self.stages)-1:
                self.downsamples.append(nn.Identity())
            else:
                self.downsamples.append(nn.Conv2d(c, channels[i+1], 2, 2, 0))

    def calclate_logit_and_feature_matching(self, fake_x, real_x):
        real_x.requires_grad = False
        fake_x = self.input_layer(fake_x)
        real_x = self.input_layer(real_x)
        feat_loss = 0
        for a, b in zip(self.stages, self.downsamples):
            fake_x = a(fake_x)
            real_x = a(real_x)
            fake_x = b(fake_x)
            fake_x = b(fake_x)
            feat_loss += (fake_x-real_x).abs().mean()
        logit = self.output_layer(fake_x)
        return logit, feat_loss

    def calclate_logit(self, fake_x):
        fake_x = self.input_layer(fake_x)
        feat_loss = 0
        for a, b in zip(self.tages, self.downsamples):
            fake_x = a(fake_x)
            fake_x = b(fake_x)
        logit = self.output_layer(fake_x)
        return logit

