from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F


class MultiHeadCrossAttention(nn.Module):
    """
    Transformer-style Multi-Head Cross Attention
    used to replace original CAM in Few-Shot CAN.
    """
    def __init__(self, dim=512, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        #Linear projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        #Output projection
        self.out_proj = nn.Linear(dim, dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, f1, f2):
        """
        f1 : [B, n1, C, H, W]  (support)
        f2 : [B, n2, C, H, W]  (query)
        """
        B, n1, C, H, W = f1.size()
        n2 = f2.size(1)

        f1 = f1.view(B, n1, C, H*W)
        f2 = f2.view(B, n2, C, H*W)

        #Mean pool
        f1 = f1.mean(-1)  # [B, n1, C]
        f2 = f2.mean(-1)  # [B, n2, C]

        # Q from query
        # K,V from support
        Q = self.q_proj(f2)
        K = self.k_proj(f1)
        V = self.v_proj(f1)

        #Multi-head reshape
        Q = Q.view(B, n2, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(B, n1, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(B, n1, self.num_heads, self.head_dim).transpose(1,2)

        #Attention QK^T
        att = torch.matmul(Q, K.transpose(-1,-2)) * self.scale
        att = F.softmax(att, dim=-1)

        out = torch.matmul(att, V)    #[B, heads, n2, head_dim]
        out = out.transpose(1,2).contiguous().view(B, n2, C)
        out = self.out_proj(out)      #[B, n2, C]

        #Expand back to [B, n2, C, H, W]
        out = out.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,H,W)
        f1_out = f1.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,H,W)
        return f1_out, out
