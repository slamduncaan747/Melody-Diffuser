from torch import nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import math
import random

@dataclass
class Config:
    vocab_size: int
    T: int
    dim: int
    epochs: int
    val_interval: int
    checkpoint_interval: int
    

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

def add_cond_noise(cond, cond_vocab=8, prob=0.1):
    mask = torch.rand_like(cond.float()) < prob
    random_gestures = torch.randint(0, cond_vocab, cond.shape, device=cond.device)
    return torch.where(mask, random_gestures, cond)

def get_loss(model, noisy_input, x0, betas, vocab_size, t, cond=None):
    if random.random() < .15:
        logits = model(noisy_input, t, None)
    else:
        logits = model(noisy_input, t, cond)
    return F.cross_entropy(logits.transpose(1,2), x0)



