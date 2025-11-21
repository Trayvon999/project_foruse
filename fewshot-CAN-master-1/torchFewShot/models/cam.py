from __future__ import absolute_import, division
import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadCrossAttention(nn.Module):
    """
    Correct Few-Shot Cross-Attention:
    - Query attends to Support  → update ftest
    - Support attends to Query  → update ftrain
    - Preserve n1=5, n2=75 order
    - Preserve spatial dims H,W
    """

    def __init__(self, dim=512, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        # Projections for support f1
        self.k1 = nn.Linear(dim, dim)
        self.v1 = nn.Linear(dim, dim)

        # Projections for query f2
        self.q2 = nn.Linear(dim, dim)
        self.k2 = nn.Linear(dim, dim)
        self.v2 = nn.Linear(dim, dim)

        self.out1 = nn.Linear(dim, dim)
        self.out2 = nn.Linear(dim, dim)

        self.scale = self.head_dim ** -0.5

    def split_heads(self, x, B, N, L):
        """[B,N,L,C] -> [B,heads,N*L,dim]"""
        return x.view(B, N * L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def merge_heads(self, x, B, N, L):
        """[B,heads,N*L,dim] -> [B,N,L,C]"""
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, N, L, self.dim)

    def forward(self, f1, f2):
        """
        f1: support [B,5,C,H,W]
        f2: query   [B,75,C,H,W]
        """

        B, n1, C, H, W = f1.shape
        n2 = f2.size(1)
        L = H * W

        f1t = f1.view(B, n1, C, L).permute(0, 1, 3, 2)  # [B,5,L,C]
        f2t = f2.view(B, n2, C, L).permute(0, 1, 3, 2)  # [B,75,L,C]

        # support → K1,V1
        K1 = self.split_heads(self.k1(f1t), B, n1, L)
        V1 = self.split_heads(self.v1(f1t), B, n1, L)

        # query → Q2,K2,V2
        Q2 = self.split_heads(self.q2(f2t), B, n2, L)
        K2 = self.split_heads(self.k2(f2t), B, n2, L)
        V2 = self.split_heads(self.v2(f2t), B, n2, L)

        # ======================
        # update ftest: Q2 @ K1
        # ======================
        att_q = torch.matmul(Q2, K1.transpose(-1, -2)) * self.scale
        att_q = F.softmax(att_q, dim=-1)

        out2 = torch.matmul(att_q, V1)
        out2 = self.merge_heads(out2, B, n2, L)
        out2 = self.out2(out2)

        out2 = out2.permute(0, 1, 3, 2).contiguous().view(B, n2, C, H, W)

        # ======================
        # update ftrain: K1 attends to Q2
        # ======================
        att_s = torch.matmul(K1, Q2.transpose(-1, -2)) * self.scale
        att_s = F.softmax(att_s, dim=-1)

        out1 = torch.matmul(att_s, V2)
        out1 = self.merge_heads(out1, B, n1, L)
        out1 = self.out1(out1)

        out1 = out1.permute(0, 1, 3, 2).contiguous().view(B, n1, C, H, W)

        return out1, out2
