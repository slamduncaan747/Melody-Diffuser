from torch import nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import math

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

def get_betas(beta0, betaT, T):
    return torch.linspace(beta0, betaT, T)

def add_noise(x0, beta_cum, vocab_size):
    rand = torch.randint(0, vocab_size, x0.shape, device=x0.device)
    mask = torch.rand(x0.shape, device=x0.device) < beta_cum.view(-1, 1)
    return torch.where(mask, rand, x0)

def get_loss(model, x0, cond, betas, vocab_size):
    B, _ = x0.shape
    T = len(betas)
    t = torch.randint(1, T+1, (B,), device=x0.device)
    beta_cum = 1-torch.cumprod(1-betas, dim=0)[t-1]

    xt = add_noise(x0, beta_cum, vocab_size)
    logits = model(xt, t, cond)
    return F.cross_entropy(logits.transpose(1,2), x0)



