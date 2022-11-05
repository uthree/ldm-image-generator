import torch
import torch.nn as nn
import math

class PositionalEncoding2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.div_term = 100

    def forward(self, x):
        ev = torch.arange(x.shape[2], device=x.device, dtype=x.dtype).reshape(1, 1, x.shape[2], 1) / x.shape[2]
        eh = torch.arange(x.shape[3], device=x.device, dtype=x.dtype).reshape(1, 1, 1, x.shape[3]) / x.shape[3]
        factors = 2 ** torch.arange(self.channels//4, device=x.device).reshape(1, self.channels//4, 1, 1)
        factors = factors / self.div_term
        ev = torch.cat([torch.sin(ev * math.pi * factors), torch.cos(ev * math.pi * factors)], dim=1)
        eh = torch.cat([torch.sin(eh * math.pi * factors), torch.cos(eh * math.pi * factors)], dim=1)
        emb = torch.cat([torch.repeat_interleave(ev, x.shape[3], dim=3), torch.repeat_interleave(eh, x.shape[2], dim=2)], dim=1)
        return x + emb

class TimeEncoding2d(nn.Module):
    def __init__(self, channels, max_timesteps=10000):
        super().__init__()
        self.channels = channels
        self.div_term = max_timesteps
    
    # t: [batch_size]
    def forward(self, x, t):
        emb = t.unsqueeze(1).expand(t.shape[0], self.channels).unsqueeze(-1).unsqueeze(-1)
        e1, e2 = torch.chunk(emb, 2, dim=1)
        factors = 2 ** torch.arange(self.channels//2, device=x.device) 
        factors = factors.unsqueeze(0).unsqueeze(2).unsqueeze(3) / self.div_term
        e1 = torch.sin(e1 * math.pi * factors)
        e2 = torch.cos(e2 * math.pi * factors)
        emb = torch.cat([e1, e2], dim=1)
        return x + emb

