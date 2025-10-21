from torch import nn
import torch

def get_betas(beta0, betaT, T):
    return torch.linspace(beta0, betaT, T)

def add_noise(x0, beta_cum, vocab_size):
    rand = torch.randint(0, vocab_size, x0.shape, device=x0.device)
    mask = torch.rand(x0.shape, device=x0.device) < beta_cum
    return torch.where(mask, rand, x0)

