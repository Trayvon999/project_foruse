from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-Attention for Few-Shot Learning
    Correct patch-level Transformer attention.
    """

    def __init__(self, dim=512, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj_q = nn.Linear(dim, dim)
        self.out_proj_k = nn.Linear(dim, dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, f1, f2):
        """
        f1 : [B, n1, C, H, W]  support
        f2 : [B, n2, C, H, W]  query
        """

        B, n1, C, H, W = f1.size()
        n2 = f2.size(1)
        HW = H * W

        # ================
        # reshape to patch tokens
        # ================
        f1 = f1.view(B, n1, C, HW).permute(0, 1, 3, 2)   # [B, n1, HW, C]
        f2 = f2.view(B, n2, C, HW).permute(0, 1, 3, 2)   # [B, n2, HW, C]

        # merge shots into a single sequence
        f1_tok = f1.reshape(B, n1 * HW, C)  # [B, n1*HW, C]
        f2_tok = f2.reshape(B, n2 * HW, C)  # [B, n2*HW, C]

        # ================
        # Linear projections
        # ================
        Q = self.q_proj(f2_tok)   # [B, n2*HW, C]
        K = self.k_proj(f1_tok)   # [B, n1*HW, C]
        V = self.v_proj(f1_tok)   # [B, n1*HW, C]

        # ================
        # reshape to multihead
        # ================
        def split_heads(x, B, L):
            return x.view(B, L, self.num_heads, self.head_dim)\
                    .permute(0, 2, 1, 3)

        Q = split_heads(Q, B, n2 * HW)  # [B, heads, Lq, d]
        K = split_heads(K, B, n1 * HW)  # [B, heads, Lk, d]
        V = split_heads(V, B, n1 * HW)  # [B, heads, Lk, d]

        # ================
        # attention QK^T
        # ================
        att = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        att = F.softmax(att, dim=-1)

        # ================
        # output
        # ================
        out2 = torch.matmul(att, V)  # [B, heads, Lq, d]

        # merge heads
        out2 = out2.permute(0, 2, 1, 3).contiguous()
        out2 = out2.view(B, n2 * HW, C)
        out2 = self.out_proj_q(out2)

        # ================
        # reshape back
        # ================
        out2 = out2.view(B, n2, HW, C).permute(0, 1, 3, 2)
        out2 = out2.view(B, n2, C, H, W)

        # symmetrical update for f1
        att_T = att.transpose(-1, -2)
        out1 = torch.matmul(att_T, Q)

        out1 = out1.permute(0, 2, 1, 3).contiguous()
        out1 = out1.view(B, n1 * HW, C)
        out1 = self.out_proj_k(out1)
        out1 = out1.view(B, n1, HW, C).permute(0, 1, 3, 2)
        out1 = out1.view(B, n1, C, H, W)

        return out1, out2
