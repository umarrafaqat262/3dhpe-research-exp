import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableAdjacency(nn.Module):
    def __init__(self, num_joints, dim):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(num_joints, num_joints))
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.uniform_(self.W, -0.01, 0.01)

    def forward(self, x):
        B, J, D = x.shape
        W_norm = F.softmax(self.W, dim=-1)
        adj_out = self.proj(torch.einsum('ij,bjd->bid', W_norm, x))
        return x + adj_out
