import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: Implement a GAN model

# class SelfAttention(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.channels = channels
#         self.mha = nn.MultiheadAttention(channels, 8, batch_first=True)
#         self.ln = nn.LayerNorm([channels])
#         self.ff_self = nn.Sequential(
#             nn.LayerNorm([channels]),
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.Linear(channels, channels),
#         )
# 
#     def forward(self, x):
#         size = x.shape[-2:]
#         x = x.view(-1, self.channels, size[0] * size[1]).swapaxes(1, 2)
#         x_ln = self.ln(x)
#         attention_value, _ = self.mha(x_ln, x_ln, x_ln)
#         attention_value = attention_value + x
#         attention_value = self.ff_self(attention_value) + attention_value
#         return attention_value.swapaxes(2, 1).view(-1, self.channels, size[0], size[1])


class SelfAttention(nn.Module):
    """Lightweight spatial self-attention (single-head)."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W)                # (B, Cq, N)
        k = self.key(x).view(B, -1, H * W)                  # (B, Cq, N)
        v = self.value(x).view(B, -1, H * W)                # (B, C, N)

        attn = torch.bmm(q.permute(0, 2, 1), k)             # (B, N, N)
        attn = F.softmax(attn / (q.size(1) ** 0.5), dim=-1) # spatial attention
        out = torch.bmm(v, attn.permute(0, 2, 1))           # (B, C, N)
        out = out.view(B, C, H, W)
        return self.gamma * out + x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class ResidualBlock(nn.Module):
    """Two convs with instance norm + skip connection."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.match_channels = (in_channels != out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True)
        )
        if self.match_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        res = self.res_conv(x) if self.match_channels else x
        return self.relu(out + res)