from torch import nn
import torch
import torch.nn.functional as F
from diffusion_utils import get_betas, add_noise, get_loss, time_encoder, Config

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-8
        self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)

        return (x/rms) * self.gamma

class SwiGLU(nn.Module):
    def __init__(self, dim, inner_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inner_dim*2)
        self.w2 = nn.Linear(inner_dim, dim)
    def forward(self, x):
        a, b = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_inner_dim, dropout=.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout = dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_inner_dim)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x):
        hidden = self.norm1(x)
        attn_out, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        x = x + self.drop1(attn_out)

        hidden = self.norm2(x)
        swiglu_out = self.ffn(hidden)
        x = x + self.drop2(swiglu_out)
        return x
    

class MelodyDiffusor(nn.Module):
    def __init__(self, vocab_size, seq_len, dim, n_layers, n_heads, ffn_inner_dim, dropout=.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        self.pos_embeddings = nn.Embedding(seq_len, dim)

        self.t_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_inner_dim, dropout=dropout) for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.token_embeddings.weight

    def forward(self, x, t, cond=None):
        B, L = x.shape
        token_emb = self.token_embeddings(x)
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_embeddings(pos_ids)

        t_emb = time_encoder(t, token_emb.size(-1))
        t_emb = self.t_proj(t_emb).unsqueeze(1)
        h = token_emb + pos_emb + t_emb

        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        logits = self.head(h)
        return logits
    

