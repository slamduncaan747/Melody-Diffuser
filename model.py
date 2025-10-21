from torch import nn
import torch
import torch.nn.functional as F
from diffusion_utils import get_betas, add_noise, get_loss
import math
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int
    T: int
    dim: int

def time_encoder(t, dim):
    half_dim = dim // 2
    period = 10000

    freqs = torch.exp(-math.log(period) * torch.arange(0, half_dim, device=t.device).float() / half_dim)

    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)

    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    return embedding

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-8
        self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
        return (x/rms) * self.gamma

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim*2)
        self.w2 = nn.Linear(dim, dim*2)
        self.w3 = nn.Linear(dim*2, dim)
    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        return self.w3(F.silu(x1) * x2)