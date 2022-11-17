import torch
import torch.nn as nn
from sinusoidal import TimeEncoding2d, PositionalEncoding2d

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
class ConvFF(nn.Module):
    def __init__(self, channels, ffn_mul=4):
        super().__init__()
        self.a = nn.Conv2d(channels, channels*ffn_mul, 1, 1, 0)
        self.b = nn.Conv2d(channels, channels*ffn_mul, 1, 1, 0)
        self.act = nn.ReLU()
        self.c = nn.Conv2d(channels*ffn_mul, channels, 1, 1, 0)
    def forward(self, x):
        return self.c(self.a(x) * self.act(self.b(x)))

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, d_model))
        self.eps = eps

    def forward(self, x):
        x = (x - x.mean(dim=2, keepdim=True)) / torch.sqrt(x.var(dim=2, keepdim=True) + self.eps)
        x = x * self.scale
        return x

class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        x = (x - x.mean(dim=1, keepdim=True)) / torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
        x = x * self.scale
        return x

class ConvBlock(nn.Module):
    def __init__(self, channels, head_dim=32):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels//head_dim)
        self.ff = ConvFF(channels)
        self.norm = ChannelNorm(channels)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.conv(x) + self.ff(x)
        x = res + x
        return x

class ConvStack(nn.Module):
    def __init__(self, channels, num_layers, head_dim=32):
        super().__init__()
        self.encodings = Encodings(channels)
        self.layers = nn.ModuleList([ConvBlock(channels, head_dim=head_dim) for _ in range(num_layers)])
    def forward(self, x, time):
        for l in self.layers:
            x = self.encodings(x, time)
            x = l(x)
        return x

class ViTLayer(nn.Module):
    def __init__(self, d_model, head_dim=64):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, d_model//head_dim, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, d_model//head_dim, batch_first=True)
        self.norm = LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, keys=None):
        res = x
        x = self.norm(x)
        attn_out, _ = self.self_attention(x, x, x)
        if keys != None:
            # Apply cross attention
            attn_out, _ = self.self_attention(x, keys, keys)
        x = attn_out + self.ff(x)
        x = res + x
        return x

class ViTStack(nn.Module):
    def __init__(self, channels, num_layers, head_dim=64):
        super().__init__()
        self.encodings = Encodings(channels)
        self.layers = nn.ModuleList([ViTLayer(channels, head_dim=head_dim) for _ in range(num_layers)])
    def forward(self, x, time, keys):
        x = self.encodings(x, time)
        # reshape to sequence
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1) # N, C, L
        x = x.transpose(1, 2) # N, L, C
        for l in self.layers:
            x = l(x)
        # reshape to image
        x = x.transpose(1, 2) # N, C, L
        x = x.reshape(*shape)
        return x

class UNetBlock(nn.Module):
    def __init__(self, stage, ch_conv):
        super().__init__()
        self.stage = stage
        self.ch_conv = ch_conv

class UNet(nn.Module):
    def __init__(self, input_channels=3, stages=[2, 2, 2, 2], channels=[64, 128, 256, 512], stem_size=1, num_mid_layers=6):
        super().__init__()
        self.encoder_first = nn.Conv2d(input_channels, channels[0], stem_size, stem_size, 0)
        self.decoder_last = nn.ConvTranspose2d(channels[0], input_channels, stem_size, stem_size, 0)
        self.mid_layers = ViTStack(channels[-1], num_mid_layers)
        self.encoder_stages = nn.ModuleList([])
        self.decoder_stages = nn.ModuleList([])
        for i, (l, c) in enumerate(zip(stages, channels)):
            enc_stage = ConvStack(c, num_layers=l)
            enc_ch_conv = nn.Identity() if i == len(stages)-1 else nn.Conv2d(channels[i], channels[i+1], 2, 2, 0)
            dec_stage = ConvStack(c, num_layers=l)
            dec_ch_conv = nn.Identity() if i == len(stages)-1 else nn.ConvTranspose2d(channels[i+1], channels[i], 2, 2, 0)
            self.encoder_stages.append(UNetBlock(enc_stage, enc_ch_conv))
            self.decoder_stages.insert(0, UNetBlock(dec_stage, dec_ch_conv))

    def forward(self, x, time, condition=None):
        x = self.encoder_first(x)
        skips = []
        for l in self.encoder_stages:
            x = l.stage(x, time=time)
            skips.insert(0, x)
            x = l.ch_conv(x)
        x = self.mid_layers(x, time=time, keys=condition)
        for i, (l, s) in enumerate(zip(self.decoder_stages, skips)):
            x = l.ch_conv(x)
            x = l.stage(x + s, time=time)
        x = self.decoder_last(x)
        return x
