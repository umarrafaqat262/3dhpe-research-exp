import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadGAT(nn.Module):
    def __init__(self, embed_dim, num_joints=17, out_dim=3):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.W_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = x + out
        out = self.proj(out)
        return out
