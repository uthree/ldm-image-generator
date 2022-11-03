import torch
import torch.nn as nn
import math
import random
from tqdm import tqdm

class PositionalEncoding2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        ev = torch.arange(x.shape[2], device=x.device, dtype=x.dtype).reshape(1, 1, x.shape[2], 1) / x.shape[2]
        eh = torch.arange(x.shape[3], device=x.device, dtype=x.dtype).reshape(1, 1, 1, x.shape[3]) / x.shape[3]
        factors = 2 ** torch.arange(self.channels//4, device=x.device).reshape(1, self.channels//4, 1, 1)
        factors = factors / 10000
        ev = torch.cat([torch.sin(ev * math.pi * factors), torch.cos(ev * math.pi * factors)], dim=1)
        eh = torch.cat([torch.sin(eh * math.pi * factors), torch.cos(eh * math.pi * factors)], dim=1)
        emb = torch.cat([torch.repeat_interleave(ev, x.shape[3], dim=3), torch.repeat_interleave(eh, x.shape[2], dim=2)], dim=1)
        return x + emb

class TimeEncoding2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x, t):
        emb = torch.full(x.shape, t, device=x.device)
        e1, e2 = torch.chunk(emb, 2, dim=1)
        factors = 2 ** torch.arange(self.channels//2, device=x.device) 
        factors = factors.unsqueeze(0).unsqueeze(2).unsqueeze(3) / 10000
        e1 = torch.sin(e1 * math.pi * factors)
        e2 = torch.cos(e2 * math.pi * factors)
        emb = torch.cat([e1, e2], dim=1)
        return x + t

class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(ChannelNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    def forward(self, x): # x: [N, C, H, W]
        m = x.mean(dim=1, keepdim=True)
        s = ((x - m) ** 2).mean(dim=1, keepdim=True)
        x = (x - m) * torch.rsqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, dim_ffn=None, kernel_size=7):
        super().__init__()
        if dim_ffn == None:
            dim_ffn = channels * 4
        self.c1 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, padding_mode='replicate', groups=channels)
        self.norm = ChannelNorm(channels)
        self.c2 = nn.Conv2d(channels, dim_ffn, 1, 1, 0)
        self.act = nn.LeakyReLU(0.1)
        self.c3 = nn.Conv2d(dim_ffn, channels, 1, 1, 0)

        self.pos_enc = PositionalEncoding2d(channels)
        self.time_enc = TimeEncoding2d(channels)

    def forward(self, x, time):
        res = x
        x = self.time_enc(x, time)
        x = self.pos_enc(x)
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = self.act(x)
        x = self.c3(x)
        return x + res

class ConvNeXtStack(nn.Module):
    def __init__(self, channels, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(ConvNeXtBlock(channels))

    def forward(self, x, time):
        for b in self.blocks:
            x = b(x, time)
        return x

class UNetBlock(nn.Module):
    def __init__(self, stage, ch_conv):
        super().__init__()
        self.stage = stage
        self.ch_conv = ch_conv

# UNet with style
class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, stages=[3, 3, 3, 3], channels=[64, 128, 256, 512], tanh=False, stem_size=4):
        super().__init__()
        self.encoder_first = nn.Conv2d(input_channels, channels[0], stem_size, stem_size, 0)

        self.decoder_last = nn.ConvTranspose2d(channels[0], output_channels, stem_size, stem_size, 0)

        self.tanh = nn.Tanh() if tanh else nn.Identity()
        self.encoder_stages = nn.ModuleList([])
        self.decoder_stages = nn.ModuleList([])
        for i, (l, c) in enumerate(zip(stages, channels)):
            enc_stage = ConvNeXtStack(c, l)
            enc_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Sequential(nn.Conv2d(channels[i], channels[i+1], 2, 2, 0), ChannelNorm(channels[i+1]))
            dec_stage = ConvNeXtStack(c, l)
            dec_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Sequential(nn.ConvTranspose2d(channels[i+1], channels[i], 2, 2, 0), ChannelNorm(channels[i]))
            self.encoder_stages.append(UNetBlock(enc_stage, enc_ch_conv))
            self.decoder_stages.insert(0, UNetBlock(dec_stage, dec_ch_conv))

    def forward(self, x, condition=None, time=None):
        x = self.encoder_first(x)
        skips = []
        for l in self.encoder_stages:
            x = l.stage(x, time)
            skips.insert(0, x)
            x = l.ch_conv(x)
        for i, (l, s) in enumerate(zip(self.decoder_stages, skips)):
            x = l.ch_conv(x)
            x = l.stage(x + s, time)
        x = self.decoder_last(x)
        x = self.tanh(x)
        return x

class DDPM(nn.Module):
    # model: model(x: Torch.tensor, condition: Torch.tensor, time: float[0 ~ 1]) -> Torch.tensor
    # x: Gaussian noise
    # condition: Condition vectors(Tokens) or None
    # time: Timestep
    def __init__(self, model=UNet(), num_timesteps=1000, beta_max=0.02, beta_min=1e-4):
        super().__init__()
        self.model = model
        self.num_timesteps = 1000
        self.beta_max = beta_max
        self.beta_min = beta_min

    def caluclate_loss(self, x, condition=None):
        t = random.randint(0, self.num_timesteps)
        t_scaled = t / self.num_timesteps
        beta = torch.linspace(self.beta_min, self.beta_max, steps=self.num_timesteps)
        alpha = 1 - beta
        alpha_bar_t = torch.prod(alpha[0:t]).item()
        noise = torch.randn(*x.shape, device=x.device)
        out = self.model(x=(math.sqrt(alpha_bar_t)*x + math.sqrt(1-alpha_bar_t) * noise), time=t_scaled, condition=condition)
        loss = torch.abs(out - noise).mean()
        return loss
    @torch.no_grad()
    def sample(self, x_shape=(1, 3, 64, 64), condition=None):
        x = torch.randn(*x_shape, device=self.model.parameters().__next__().device)
        beta = torch.linspace(self.beta_min, self.beta_max, steps=self.num_timesteps)
        alpha = 1 - beta
        bar = tqdm(total=self.num_timesteps)
        for t in reversed(range(1, self.num_timesteps)):
            k = 1 / math.sqrt(alpha[t])
            z = torch.randn(*x.shape, device=x.device)
            sigma = 1 - k
            if t == 1:
                sigma = 0
            alpha_bar_t = torch.prod(alpha[0:t]).item()
            x = k * (x - (1-alpha[t])/math.sqrt(1-alpha_bar_t) * self.model(x=x, time=t))
            bar.update()
        return x
