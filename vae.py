import torch
import torch.nn as nn
import torch.optim as optim

class ResBlock(nn.Module):
    def __init__(self, channels, dim_group=32, channel_mul=4):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels//dim_group)
        self.c2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.act(x)
        x = self.c2(x)
        return x + res

class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_channels=3, channels=[64, 128, 256], stages=[2, 2, 4]):
        super().__init__()
        self.stacks = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])
        self.input_conv = nn.Conv2d(input_channels, channels[0], 1, 1, 0)
        self.output_conv = nn.Conv2d(channels[-1], latent_channels*2, 1, 1) # to mean, logvar
        for i, (c, s) in enumerate(zip(channels, stages)):
            self.stacks.append(nn.Sequential(*[ResBlock(c) for _ in range(s)]))
            if i != len(channels)-1:
                self.downsamples.append(nn.Conv2d(channels[i], channels[i+1], 2, 2, 0))
            else:
                self.downsamples.append(nn.Identity())

    def forward(self, x):
        # encode
        x = self.input_conv(x)
        for s, d in zip(self.stacks, self.downsamples):
            x = s(x)
            x = d(x)
        x = self.output_conv(x)
        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, output_channels=3, latent_channels=3, channels=[256, 128, 64], stages=[9, 3, 3]):
        super().__init__()
        self.stacks = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])
        self.input_conv = nn.Conv2d(latent_channels, channels[0], 1, 1, 0)
        self.output_conv = nn.Conv2d(channels[-1], output_channels, 1, 1, 0)
        for i, (c, s) in enumerate(zip(channels, stages)):
            self.stacks.append(nn.Sequential(*[ResBlock(c) for _ in range(s)]))
            if i != len(channels)-1:
                self.upsamples.append(nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2, 0))
            else:
                self.upsamples.append(nn.Identity())
    
    def forward(self, x):
        x = self.input_conv(x)
        for s, u in zip(self.stacks, self.upsamples):
            x = s(x)
            x = u(x)
        x = self.output_conv(x)
        return x

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def caluclate_loss(self, x):
        mean, logvar = self.encoder(x) 
        gaussian = torch.randn(*mean.shape, device=x.device)
        z = gaussian * torch.exp(logvar) + mean
        y = self.decoder(z)
        reconstruction_loss = torch.abs(x - y).mean()
        kl_divergence_loss = -1 - logvar.mean() + torch.exp(logvar).mean() + (mean**2).mean()
        return reconstruction_loss, kl_divergence_loss, y

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, channels=[64, 128, 256], stages=[2, 2, 4]):
        super().__init__()
        self.stacks = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])
        self.input_conv = nn.Conv2d(input_channels, channels[0], 1, 1, 0)
        self.to_logit = nn.Conv2d(channels[-1], 1, 1, 1, 0)
        for i, (c, s) in enumerate(zip(channels, stages)):
            self.stacks.append(nn.Sequential(*[ResBlock(c) for _ in range(s)]))
            if i != len(channels)-1:
                self.downsamples.append(nn.Conv2d(channels[i], channels[i+1], 2, 2, 0))
            else:
                self.downsamples.append(nn.Identity())

    def forward(self, x):
        x = self.input_conv(x)
        for s, d in zip(self.stacks, self.downsamples):
            x = s(x)
            x = d(x)
        x = self.to_logit(x)
        return x

