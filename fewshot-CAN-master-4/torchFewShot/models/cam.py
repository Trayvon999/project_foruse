from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.mlp = nn.Sequential(
            nn.Conv2d(36, 48, 1),
            nn.ReLU(),
            nn.Conv2d(48, 36, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def get_attention(self, a):
        """
        输入: a.shape = [B, n1, n2, HW]
        输出: 注意力权重 a.shape = [B, n1, n2, HW]
        """

        input_a = a
        a = a.mean(3)
        a = a.transpose(1, 3)   # [B, n2, n1, HW] → [B, HW, n1, n2]

        a = self.mlp(a)
        a = a.transpose(1, 3)
        a = a.unsqueeze(3)   # [B, n1, n2, 1, HW]


        #LayerNorm Attention
        norm = input_a
        norm = norm - norm.mean(dim=-1, keepdim=True)
        norm = norm / (norm.std(dim=-1, keepdim=True) + 1e-6)

        a = torch.mean(norm * a, -1)

        a = F.softmax(a / 0.025, dim=-1) + 1
        return a


    def forward(self, f1, f2):
        """
        f1: [B, n1, C, H, W]
        f2: [B, n2, C, H, W]
        """

        b, n1, c, h, w = f1.size()
        n2 = f2.size(1)

        f1 = f1.view(b, n1, c, -1)   # [B, n1, C, HW]
        f2 = f2.view(b, n2, c, -1)

        # L2 norm
        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)

        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2)  # [B, n1, 1, HW, C]
        f2_norm = f2_norm.unsqueeze(1)                  # [B, 1, n2, C, HW]

        a1_raw = torch.matmul(f1_norm, f2_norm)     # [B, n1, n2, HW, HW]
        a2_raw = a1_raw.transpose(3, 4)

        a1 = self.get_attention(a1_raw)
        a2 = self.get_attention(a2_raw)

        alpha = torch.sigmoid(self.alpha)
        A = alpha * a1 + (1 - alpha) * a2


        #Residual Attention
        f1_weighted = f1.unsqueeze(2) * A.unsqueeze(3)
        f2_weighted = f2.unsqueeze(1) * A.unsqueeze(3)

        f1 = f1_weighted + f1.unsqueeze(2)
        f2 = f2_weighted + f2.unsqueeze(1)
        f1 = f1.view(b, n1, n2, c, h, w)
        f2 = f2.view(b, n1, n2, c, h, w)

        return f1.transpose(1, 2), f2.transpose(1, 2)

