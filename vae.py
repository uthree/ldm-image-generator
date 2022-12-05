import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ChannelNorm


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=8192, dim=4):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, dim))

    def calculate_loss(self, x):
        e = self.embed(self.quantize(x))
        reg_loss = F.l1_loss(x, e.detach())
        embedding_loss = F.l1_loss(e, x.detach())
        return embedding_loss + reg_loss

    @torch.no_grad()
    def quantize(self, x):
        prob = torch.matmul(x, self.embeddings.transpose(0, 1))
        indexes = torch.argmax(prob, dim=2)
        return indexes

    def embed(self, x):
        out = F.embedding(x, self.embeddings)
        return out


class VAE(nn.Module):
    def __init__(self, encoder, decoder, quantizer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer

    def calclate_loss(self, x, noise_gain=0.1):
        z = self.encoder(x) + torch.randn(x.shape, x.device) * noise_gain
        loss_reg = self.quantizer.calculate_loss(
                z.reshape(z.shape[0], z.shape[1], -1).transpose(1,2))
        y = self.decoder(z)
        loss_recon = (x.detach()-y).abs().mean()
        return loss_recon, loss_reg, y
    
    @torch.no_grad()
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    @ torch.no_grad()
    def decode(self, z):
        return self.decoder(z)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2)
        self.c2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        return self.c2(self.act(self.c1(x))) + x

class ResStack(nn.Module):
    def __init__(self, channels, num_layers=2):
        super().__init__()
        self.seq = nn.Sequential(*[ResBlock(channels) for _ in range(num_layers)])

    def forward(self, x):
        return self.seq(x)

class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_channels=4, channels=[64, 128, 256, 512], stages=[2, 2, 2, 2]):
        super().__init__()
        self.input_layer = nn.Conv2d(input_channels, channels[0], 1, 1, 0)
        self.output_layer = nn.Conv2d(channels[-1], latent_channels, 1, 1, 0)
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
        return self.output_layer(x)

class DecoderBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 7, 1, 3)
        self.norm = ChannelNorm(channels)
        self.c2  = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.LeakyReLU(0.2)
        self.c3 = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = self.act(x)
        x = self.c3(x)
        return x + res

class DecoderStack(nn.Module):
    def __init__(self, channels, num_layers, output_channels=3):
        super().__init__()
        self.layers = nn.Sequential(*[DecoderBlock(channels) for _ in range(num_layers)])
        self.to_rgb = nn.Conv2d(channels, output_channels, 1, 1, 0)

    def forward(self, x):
        x = self.layers(x)
        return x, self.to_rgb(x)

class Decoder(nn.Module):
    def __init__(self, output_channels=3, latent_channels=4, channels=[512, 256, 128, 64], stages=[2, 2, 2, 2]):
        super().__init__()
        self.input_layer = nn.Conv2d(latent_channels, channels[0], 1, 1, 0)
        self.output_layer = nn.Conv2d(channels[-1], output_channels, 1, 1, 0)
        self.stages = nn.ModuleList([DecoderStack(c, l, output_channels=output_channels) for c, l in zip(channels, stages)])
        self.upsamples = nn.ModuleList([])
        for i, c in enumerate(channels):
            if i == 0:
                self.upsamples.append(nn.Identity())
            else:
                self.upsamples.append(nn.ConvTranspose2d(channels[i-1], c, 2, 2, 0))

    def forward(self, x):
        rgb_out = None
        x = self.input_layer(x)
        for a, b in zip(self.upsamples, self.stages):
            x = a(x)
            x, rgb = b(x)
            if rgb_out == None:
                rgb_out = rgb
            else:
                rgb_out = F.interpolate(rgb_out, scale_factor=2) + rgb
        return rgb_out

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, channels=[32, 48, 48, 96], stages=[2, 2, 2, 2], stem_size=1):
        super().__init__()
        self.input_layer = nn.Conv2d(input_channels, channels[0], stem_size, stem_size, 0)
        self.stages = nn.ModuleList([ResStack(c, l) for c, l in zip(channels, stages)])
        self.early_exits = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])
        for i, c in enumerate(channels):
            if i == len(self.stages)-1:
                self.downsamples.append(nn.Identity())
            else:
                self.downsamples.append(nn.Conv2d(c, channels[i+1], 2, 2, 0))
            self.early_exits.append(nn.Conv2d(c, 1, 1, 1, 0))

    def calclate_logit_and_feature_matching(self, fake_x, real_x):
        real_x.requires_grad = False
        fake_x = self.input_layer(fake_x)
        real_x = self.input_layer(real_x)
        logit = 0
        feat_loss = 0
        for a, b, c in zip(self.stages, self.downsamples, self.early_exits):
            fake_x = a(fake_x)
            real_x = a(real_x)
            feat_loss += (fake_x-real_x).abs().mean()
            logit = logit + c(fake_x).mean()
            fake_x = b(fake_x)
            real_x = b(real_x)
        return logit, feat_loss

    def calclate_logit(self, fake_x):
        fake_x = self.input_layer(fake_x)
        feat_loss = 0
        logit = 0
        for a, b, c in zip(self.stages, self.downsamples, self.early_exits):
            fake_x = a(fake_x)
            logit = logit + c(fake_x).mean()
            fake_x = b(fake_x)
        return logit

