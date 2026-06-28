import torch
import torch.nn as nn
import torch.nn.functional as F

HYPEREDGES = [
    [0, 1, 2, 3],
    [0, 4, 5, 6],
    [0, 7, 8],
    [8, 9, 10],
    [8, 11, 12, 13],
    [8, 14, 15, 16],
]

class HyperGCN(nn.Module):
    def __init__(self, dim, num_joints=17):
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints

        inc = torch.zeros(len(HYPEREDGES), num_joints)
        for e, joints in enumerate(HYPEREDGES):
            for j in joints:
                inc[e, j] = 1.0
        self.register_buffer('incidence', inc)

        self.edge_mlp = nn.Linear(dim, dim, bias=False)

        self.alpha = nn.Parameter(torch.ones(len(HYPEREDGES), 1, 1) * 0.5)

        self.proj = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0)

    def forward(self, x):
        B, J, D = x.shape

        edge_feats = torch.einsum('ej,bjd->bed', self.incidence, x)
        edge_feats = edge_feats / self.incidence.sum(dim=1, keepdim=True).clamp(min=1)
        edge_feats = self.edge_mlp(edge_feats)

        hyper_out = torch.einsum('ej,bed->bjd', self.incidence, edge_feats)
        hyper_out = hyper_out / self.incidence.sum(dim=0, keepdim=True).t().clamp(min=1)

        out = x + self.proj(hyper_out)

        return out
