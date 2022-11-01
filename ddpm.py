import torch
import torch.nn as nn
import math

class DDPM(nn.Module):
    # model: model(x: Torch.tensor, condition: Torch.tensor, time: float[0 ~ 1]) -> Torch.tensor
    # x: Gaussian noise
    # condition: Condition vectors(Tokens) or None
    # time: Timestep
    def __init__(self, model=nn.Identity(), num_timesteps=1000, beta_max=0.02, beta_min=10e-4):
        super().__init__()
        self.model = model
        self.num_timesteps = 1000

class PositionalEncoding2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        ev = torch.arange(x.shape[2], device=x.device, dtype=x.dtype).reshape(1, 1, x.shape[2], 1) / x.shape[2]
        eh = torch.arange(x.shape[3], device=x.device, dtype=x.dtype).reshape(1, 1, 1, x.shape[3]) / x.shape[3]
        factors = torch.arange(self.channels//4).reshape(1, self.channels//4, 1, 1)
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
        factors = 2 ** torch.arange(self.channels //2, device=x.device)
        factors = factors.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        e1 = torch.sin(e1 * math.pi * factors)
        e2 = torch.cos(e2 * math.pi * factors)
        emb = torch.cat([e1, e2], dim=1)
        return x + t

class SelfAttention(nn.Module):
    def __init__(self, dim=512, nheads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, nheads, batch_first=True)

    def forward(self, x):
        out, weight = self.attention(x, x, x)
        return out + x

class CrossAttention(nn.Module):
    def __init__(self, dim=512, nheads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, nheads, batch_first=True)
    
    def forward(self, x, c):
        out, weight = self.attention(x, c, c)
        return out + x

class FeedForward(nn.Module):
    def __init__(self, embedding_dim=512, internal_dim=2048):
        super().__init__()
        self.w1 = nn.Linear(embedding_dim, internal_dim)
        self.v1 = nn.Linear(embedding_dim, internal_dim)
        self.act = nn.LeakyReLU(0.1)
        self.w2 = nn.Linear(internal_dim, embedding_dim)
    def forward(self, x):
        self.w2(self.w1(x) * self.v1(x)) + x


