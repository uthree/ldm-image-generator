import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ReGLU
class ReGLU(nn.Module):
    def __init__(self, channels, ffn_mul=4):
        super().__init__()
        self.a = nn.Conv2d(channels, channels*ffn_mul, 1, 1, 0)
        self.b = nn.Conv2d(channels, channels*ffn_mul, 1, 1, 0)
        self.act = nn.ReLU()
        self.c = nn.Conv2d(channels*ffn_mul, channels, 1, 1, 0)
    def forward(self, x):
        return self.c(self.a(x) * self.act(self.b(x)))

# Channel Normalization
class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = (x - x.mean(dim=1, keepdim=True)) / torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
        return x

# Randomly choiced Mixture of Experts
class RandomMoE(nn.Module):
    def __init__(self, channels, ffn_mul=2, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([ReGLU(channels, ffn_mul=ffn_mul)])
    
    def forward(self, x):
        mod = random.choice(self.experts)
        return mod(x)
