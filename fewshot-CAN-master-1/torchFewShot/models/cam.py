from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadCrossAttention(nn.Module):
    """
    A corrected Transformer-style Multi-Head Cross Attention module.
    This version:
    - Keeps spatial structure (H, W)
    - Performs cross-attention between support & query at patch-level
    - Returns [B, n1, C, H, W] and [B, n2, C, H, W], matching the original CAN API
    """

    def __init__(self, dim=512, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Output projections
        self.out_proj_q = nn.Linear(dim, dim)
        self.out_proj_k = nn.Linear(dim, dim)

        self.scale = self.head_dim ** -0.5

    def _reshape_for_attention(self, x, B, N, H, W):
        """
        x: [B, N, C, H, W]
        return [B, N, H*W, C]
        """
        x = x.view(B, N, x.size(2), H * W)       # → [B, N, C, HW]
        x = x.permute(0, 1, 3, 2).contiguous()   # → [B, N, HW, C]
        return x

    def _restore_shape(self, x, B, N, H, W):
        """
        x: [B, N, HW, C]
        return [B, N, C, H, W]
        """
        x = x.permute(0, 1, 3, 2).contiguous()       # → [B, N, C, HW]
        x = x.view(B, N, self.dim, H, W)            # → [B, N, C, H, W]
        return x

    def _multihead(self, x, B, N, L):
        """
        x: [B, N, L, C]
        → reshape to multi-head attention format
        return: [B, heads, N, L, head_dim]
        """
        return x.view(B, N, L, self.num_heads, self.head_dim) \
                .permute(0, 3, 1, 2, 4) \
                .contiguous()

    def _merge_heads(self, x, B, N, L):
        """
        x: [B, heads, N, L, head_dim]
        → merge heads
        return: [B, N, L, C]
        """
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        return x.view(B, N, L, self.dim)

    def forward(self, f1, f2):
        """
        f1 : support  [B, n1, C, H, W]
        f2 : query    [B, n2, C, H, W]
        """

        B, n1, C, H, W = f1.size()
        n2 = f2.size(1)

        # Flatten spatial dims (patch-level tokens)
        f1_flat = self._reshape_for_attention(f1, B, n1, H, W)  # [B, n1, HW, C]
        f2_flat = self._reshape_for_attention(f2, B, n2, H, W)  # [B, n2, HW, C]

        HW = H * W

        # Apply Q/K/V linear projections
        Q = self.q_proj(f2_flat)   # [B, n2, HW, C]
        K = self.k_proj(f1_flat)   # [B, n1, HW, C]
        V = self.v_proj(f1_flat)   # [B, n1, HW, C]

        # Multi-head reshape
        Q = self._multihead(Q, B, n2, HW)
        K = self._multihead(K, B, n1, HW)
        V = self._multihead(V, B, n1, HW)

        # Attention: Q * K^T
        att = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        att = F.softmax(att, dim=-1)

        # Aggregate values
        out2 = torch.matmul(att, V)        # → query-updated features
        out2 = self._merge_heads(out2, B, n2, HW)
        out2 = self.out_proj_q(out2)

        # Symmetry: support also attends to query
        att_T = att.transpose(2, 3)
        out1 = torch.matmul(att_T, Q)       # → support-updated features
        out1 = self._merge_heads(out1, B, n1, HW)
        out1 = self.out_proj_k(out1)

        # Restore spatial dims back to [B, n, C, H, W]
        f1_out = self._restore_shape(out1, B, n1, H, W)
        f2_out = self._restore_shape(out2, B, n2, H, W)

        return f1_out, f2_out
