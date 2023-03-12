import torch
import torch.nn as nn
from sinusoidal import TimeEncoding2d, PositionalEncoding2d
from modules import RandomMoE, ChannelNorm
from attention import WindowAttention, CrossAttention
import random

# Time + Position encoding
class Encodings(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj1 = nn.Conv2d(channels*2, channels*4, 1, 1, 0)
        self.act = nn.ReLU()
        self.proj2 = nn.Conv2d(channels*4, channels*2, 1, 1, 0)
        self.pe = PositionalEncoding2d(channels, return_encoding_only=True)
        self.te = TimeEncoding2d(channels, return_encoding_only=True)

    def forward(self, x, t):
        embs = torch.cat([self.pe(x), self.te(x, t)], dim=1)
        embs = self.proj2(self.act(self.proj1(embs)))
        mul, bias = torch.chunk(embs, 2, dim=1)
        x = x * mul + bias
        return x

class SwinBlock(nn.Module):
    def __init__(self, channels, head_dim=32, window_size=6, shift=0, attention=True, stochastic_depth=0.25):
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.ffn = RandomMoE(channels)
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.stochastic_depth = stochastic_depth
        self.attention_flag = attention
        if attention:
            self.self_attention = WindowAttention(channels, n_heads=channels//head_dim, window_size=window_size, shift=shift)
            self.cross_attention = CrossAttention(channels, n_heads=channels//head_dim)
        self.encodings = Encodings(channels)

    def forward(self, x, t, c=None):
        if self.training and random.random() <= self.stochastic_depth:
            return x
        res = x
        x = self.norm(x)
        x = self.encodings(x, t)
        x = self.ffn(x) + self.conv(x) + (self.self_attention(x) if self.attention_flag else 0)
        if c != None and self.attention_flag:
            x = x + self.cross_attention(x, c)
        x = x + res
        return x

class SwinStack(nn.Module):
    def __init__(self, channels, head_dim=32, window_size=6, num_blocks=2, attention=True):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            shift = window_size // 2 if i % 2 == 0 else 0
            # add attention only last 2 layers
            if i >= num_blocks-2:
                flag_attn = attention
            else:
                flag_attn = False
            self.blocks.append(SwinBlock(channels, head_dim, window_size, shift, attention=flag_attn))

    def forward(self, x, t, c=None):
        for b in self.blocks:
            x = b(x, t, c)
        return x

class UNetBlock(nn.Module):
    def __init__(self, stage, ch_conv):
        super().__init__()
        self.stage = stage
        self.ch_conv = ch_conv

class UNet(nn.Module):
    def __init__(self, input_channels=8, stages=[2, 2, 8, 2], channels=[96, 192, 384, 768], stem_size=1):
        super().__init__()
        self.encoder_first = nn.Conv2d(input_channels, channels[0], stem_size, stem_size, 0)
        self.decoder_last = nn.ConvTranspose2d(channels[0], input_channels, stem_size, stem_size, 0)
        self.encoder_stages = nn.ModuleList([])
        self.decoder_stages = nn.ModuleList([])
        for i, (l, c) in enumerate(zip(stages, channels)):
            enc_stage = SwinStack(c, num_blocks=l, attention=False)
            enc_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Sequential(nn.Conv2d(channels[i], channels[i+1], 1, 1, 0), nn.AvgPool2d(kernel_size=2))
            dec_stage = SwinStack(c, num_blocks=l)
            dec_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Sequential(nn.Upsample(scale_factor=2) ,nn.Conv2d(channels[i+1], channels[i], 1, 1, 0))
            self.encoder_stages.append(UNetBlock(enc_stage, enc_ch_conv))
            self.decoder_stages.insert(0, UNetBlock(dec_stage, dec_ch_conv))

    def forward(self, x, time, condition=None):
        x = self.encoder_first(x)
        skips = []
        for i, l in enumerate(self.encoder_stages):
            x = l.stage(x, time)
            if i == len(self.encoder_stages)-1:
                skips.insert(0, 0) # Insert zero if last encoder layer
            else:
                skips.insert(0, x) 
            x = l.ch_conv(x)
        for i, (l, s) in enumerate(zip(self.decoder_stages, skips)):
            x = l.ch_conv(x)
            x = l.stage(x + s, time)
        x = self.decoder_last(x)
        return x
