import torch
import torch.nn as nn
from sinusoidal import TimeEncoding2d, PositionalEncoding2d
from attention import WindowAttention

class Encodings(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj1 = nn.Conv2d(channels*2, channels*2, 1, 1, 0)
        self.act = nn.ReLU()
        self.proj2 = nn.Conv2d(channels*2, channels*2, 1, 1, 0)
        self.pe = PositionalEncoding2d(channels, return_encoding_only=True)
        self.te = TimeEncoding2d(channels, return_encoding_only=True)

    def forward(self, x, t):
        embs = torch.cat([self.pe(x), self.te(x, t)], dim=1)
        embs = self.proj2(self.act(self.proj1(embs)))
        mul, bias = torch.chunk(embs, 2, dim=1)
        x = x * mul + bias
        return x

# FFN (ReGLU)
class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_mul=4):
        super().__init__()
        self.a = nn.Linear(d_model, d_model*ffn_mul)
        self.b = nn.Linear(d_model, d_model*ffn_mul)
        self.act = nn.ReLU()
        self.c = nn.Linear(d_model*ffn_mul, d_model)

    def forward(self, x):
        return self.c(self.a(x) * self.act(self.b(x)))

# Conv (ReGLU)
class ConvFFN(nn.Module):
    def __init__(self, channels, ffn_mul=4):
        super().__init__()
        self.a = nn.Conv2d(channels, channels*ffn_mul, 1, 1, 0)
        self.b = nn.Conv2d(channels, channels*ffn_mul, 1, 1, 0)
        self.act = nn.ReLU()
        self.c = nn.Conv2d(channels*ffn_mul, channels, 1, 1, 0)
    def forward(self, x):
        return self.c(self.a(x) * self.act(self.b(x)))

class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        x = (x - x.mean(dim=1, keepdim=True)) / torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
        x = x * self.scale
        return x

class SwinBlock(nn.Module):
    def __init__(self, channels, head_dim=32, window_size=4, shift=0):
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.ffn = ConvFFN(channels)
        self.attention = WindowAttention(channels, n_heads=channels//head_dim, window_size=window_size, shift=shift)
        self.encodings = Encodings(channels)

    def forward(self, x, t):
        x = self.encodings(x, t)
        res = x
        x = self.norm(x)
        x = self.attention(x) + self.ffn(x)
        x = x + res
        return x

class SwinStack(nn.Module):
    def __init__(self, channels, head_dim=32, window_size=4, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            shift = window_size // 2 if i % 2 == 0 else 0
            self.blocks.append(SwinBlock(channels, head_dim, window_size, shift))

    def forward(self, x, t):
        for b in self.blocks:
            x = b(x, t)
        return x

class UNetBlock(nn.Module):
    def __init__(self, stage, ch_conv):
        super().__init__()
        self.stage = stage
        self.ch_conv = ch_conv

class UNet(nn.Module):
    def __init__(self, input_channels=3, stages=[2, 2, 2, 2], channels=[64, 128, 256, 512], stem_size=1):
        super().__init__()
        self.encoder_first = nn.Conv2d(input_channels, channels[0], stem_size, stem_size, 0)
        self.decoder_last = nn.ConvTranspose2d(channels[0], input_channels, stem_size, stem_size, 0)
        self.encoder_stages = nn.ModuleList([])
        self.decoder_stages = nn.ModuleList([])
        for i, (l, c) in enumerate(zip(stages, channels)):
            enc_stage = SwinStack(c, num_blocks=l)
            enc_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Sequential(nn.Conv2d(channels[i], channels[i+1], 1, 1, 0), nn.AvgPool2d(kernel_size=2))
            dec_stage = SwinStack(c, num_blocks=l)
            dec_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Sequential(nn.Upsample(scale_factor=2) ,nn.Conv2d(channels[i+1], channels[i], 1, 1, 0))
            self.encoder_stages.append(UNetBlock(enc_stage, enc_ch_conv))
            self.decoder_stages.insert(0, UNetBlock(dec_stage, dec_ch_conv))

    def forward(self, x, time, condition=None):
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
        return x
