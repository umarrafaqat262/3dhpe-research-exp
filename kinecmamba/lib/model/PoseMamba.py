## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt
import os
import sys
current_directory = os.path.dirname(__file__) + '/../' + '../'
sys.path.append(current_directory)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from functools import partial
import torch.fft

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np

from lib.model.mambablocks import BiSTSSMBlock
from lib.model.modules.ssi import LearnableAdjacency
from lib.model.modules.msm import PerJointDelta
from lib.model.modules.hypergcn import HyperGCN
from lib.model.modules.head_gat import HeadGAT

LIMBS = [[0,1], [1,2], [2,3], [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10], [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]]

def _make_joint_to_limb(num_joints=17):
    mat = torch.zeros(num_joints, len(LIMBS))
    for j, (p, c) in enumerate(LIMBS):
        mat[p, j] = 1.0
        mat[c, j] = 1.0
    counts = mat.sum(dim=1, keepdim=True).clamp(min=1)
    return mat / counts

class  PoseMamba(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256, depth=6, mlp_ratio=2., drop_rate=0., drop_path_rate=0.2,  norm_layer=None, use_ssi=False, use_msm=False, use_hypergcn=False, use_gat_head=False, forward_type='v2_plus_poselimbs'):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3
        self.in_chans = in_chans
        self.register_buffer('joint_to_limb', _make_joint_to_limb(num_joints))
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth
        self.STEblocks = nn.ModuleList([
           BiSTSSMBlock(
                hidden_dim = embed_dim_ratio,
                mlp_ratio = mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                forward_type=forward_type
                )
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
           BiSTSSMBlock(
                hidden_dim = embed_dim,
                mlp_ratio = mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                forward_type=forward_type
                )
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.use_ssi = use_ssi
        if use_ssi:
            self.ssi = LearnableAdjacency(num_joints, embed_dim_ratio)
        self.use_msm = use_msm
        if use_msm:
            d_inner = int(2.0 * embed_dim_ratio)
            self.msm = PerJointDelta(num_joints, d_inner)
            for blk in self.STEblocks:
                blk.op.msm = self.msm
        self.use_hypergcn = use_hypergcn
        if use_hypergcn:
            self.hypergcn = HyperGCN(embed_dim, num_joints)

        if use_gat_head:
            self.head = HeadGAT(embed_dim, num_joints=num_joints, out_dim=out_dim)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim , out_dim),
            )


    def STE_forward(self, x):
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c  -> (b f) n c', )
        x = self.Spatial_patch_to_embedding(x)
        if self.use_ssi:
            x = self.ssi(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = rearrange(x, '(b f) n c  -> b f n c', f=f)
        blk = self.STEblocks[0]
        x = blk(x)
        x = self.Spatial_norm(x)
        return x

    def TTE_foward(self, x):
        b, f, n, c  = x.shape
        x = rearrange(x, 'b f n cw -> (b n) f cw', f=f)
        x += self.Temporal_pos_embed[:,:f,:]
        x = self.pos_drop(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        blk = self.TTEblocks[0]
        x = blk(x)
        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = tteblock(x)
            x = self.Temporal_norm(x)
        return x

    def compute_bone_features(self, x):
        b, f, n, c = x.shape
        xy = x[..., :2]
        limbs = xy[:, :, LIMBS, :]
        limb_vecs = limbs[:, :, :, 1, :] - limbs[:, :, :, 0, :]
        bone_feats = torch.einsum('jn,btnd->btjd', self.joint_to_limb, limb_vecs)
        return bone_feats

    def forward(self, x):
        b, f, n, c = x.shape
        if self.in_chans == 5 and c == 3:
            bone_feats = self.compute_bone_features(x)
            x = torch.cat([x, bone_feats], dim=-1)
        x = self.STE_forward(x)
        x = self.TTE_foward(x)
        x = self.ST_foward(x)
        if self.use_hypergcn:
            x = rearrange(x, 'b f n d -> (b f) n d')
            x = self.hypergcn(x)
            x = rearrange(x, '(b f) n d -> b f n d', b=b, f=f)
        if isinstance(self.head, HeadGAT):
            x = rearrange(x, 'b f n d -> (b f) n d')
            x = self.head(x)
            x = rearrange(x, '(b f) n d -> b f n d', b=b, f=f)
        else:
            x = self.head(x)
            x = x.view(b, f, n, -1)
        return x
