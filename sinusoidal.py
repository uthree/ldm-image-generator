import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding2d(nn.Module):
    def __init__(self, channels, return_encoding_only=False):
        super().__init__()
        self.channels = channels
        self.return_encoding_only = return_encoding_only

    def forward(self, x):
        ev = torch.arange(x.shape[2], device=x.device, dtype=x.dtype).reshape(1, 1, x.shape[2], 1) / x.shape[2]
        eh = torch.arange(x.shape[3], device=x.device, dtype=x.dtype).reshape(1, 1, 1, x.shape[3]) / x.shape[3]
        factors = 1 / (2 ** (torch.arange(self.channels//4, device=x.device).reshape(1, self.channels//4, 1, 1) / (self.channels//4)))
        ev = torch.cat([torch.sin(ev * math.pi * factors), torch.cos(ev * math.pi * factors)], dim=1)
        eh = torch.cat([torch.sin(eh * math.pi * factors), torch.cos(eh * math.pi * factors)], dim=1)
        emb = torch.cat([torch.repeat_interleave(ev, x.shape[3], dim=3), torch.repeat_interleave(eh, x.shape[2], dim=2)], dim=1)
        emb = emb.expand(*x.shape)
        ret = emb if self.return_encoding_only else x + emb
        return ret

class TimeEncoding2d(nn.Module):
    def __init__(self, channels, max_timesteps=10000, return_encoding_only=False):
        super().__init__()
        self.channels = channels
        self.max_timesteps = max_timesteps
        self.return_encoding_only = return_encoding_only

    # t: [batch_size]
    def forward(self, x, t):
        emb = t.unsqueeze(1).expand(t.shape[0], self.channels).unsqueeze(-1).unsqueeze(-1)
        e1, e2 = torch.chunk(emb, 2, dim=1)
        factors = 1 / (self.max_timesteps ** (torch.arange(self.channels//2, device=x.device) / (self.channels//2)))
        factors = factors.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        e1 = torch.sin(e1 * math.pi * factors)
        e2 = torch.cos(e2 * math.pi * factors)
        emb = torch.cat([e1, e2], dim=1).expand(*x.shape)

        ret = emb if self.return_encoding_only else x + emb
        return ret
