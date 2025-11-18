import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# class SelfAttention(nn.Module):
#     """
#     A simple Multi-Head Self-Attention block.
#     This is the core component of a Transformer layer.
#     """
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
# 
#         self.in_proj = nn.Linear(embed_dim, embed_dim * 3) # Q, K, V
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
# 
#     def forward(self, x):
#         # Input x: (B, N, C) where N is sequence length, C is embed_dim
#         B, N, C = x.shape
#         
#         # 1. Project to Q, K, V
#         # (B, N, C) -> (B, N, C*3) -> (B, N, 3, H, D) -> (3, B, H, N, D)
#         qkv = self.in_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2] # Each is (B, H, N, D)
# 
#         # 2. Scaled Dot-Product Attention
#         # (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
#         attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         attn_probs = F.softmax(attn_scores, dim=-1)
#         
#         # (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
#         context = attn_probs @ v
#         
#         # 3. Concat heads and project out
#         # (B, H, N, D) -> (B, N, H, D) -> (B, N, C)
#         context = context.transpose(1, 2).reshape(B, N, self.embed_dim)
#         
#         # (B, N, C) -> (B, N, C)
#         return self.out_proj(context)

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Calculate div_term based on half the channels (since we split for H and W)
        div_term = torch.exp(torch.arange(0, self.channels // 2, 2).float() * (-math.log(10000.0) / (self.channels // 2))).to(x.device)
        
        # Height encoding
        pe_h = torch.zeros(H, self.channels // 2, device=x.device)
        pos_h = torch.arange(0, H, device=x.device).unsqueeze(1)
        pe_h[:, 0::2] = torch.sin(pos_h * div_term)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term)
        
        # Expand Height: [H, C//2] -> [1, H, 1, C//2] -> [B, H, W, C//2]
        pe_h = pe_h.unsqueeze(0).unsqueeze(2).expand(B, H, W, self.channels // 2)
        pe_h = pe_h.permute(0, 3, 1, 2) # [B, C//2, H, W]

        # Width encoding
        pe_w = torch.zeros(W, self.channels // 2, device=x.device)
        pos_w = torch.arange(0, W, device=x.device).unsqueeze(1)
        pe_w[:, 0::2] = torch.sin(pos_w * div_term)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term)
        
        # Expand Width: [W, C//2] -> [1, 1, W, C//2] -> [B, H, W, C//2]
        pe_w = pe_w.unsqueeze(0).unsqueeze(1).expand(B, H, W, self.channels // 2)
        pe_w = pe_w.permute(0, 3, 1, 2) # [B, C//2, H, W]
        
        # Concatenate height and width encodings
        pe = torch.cat([pe_h, pe_w], dim=1)
        
        return x + pe

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.pos_encoding = PositionalEncoding2D(channels)
        self.mha = nn.MultiheadAttention(channels, 8, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # Add positional encoding while still in [B, C, H, W] format
        x = self.pos_encoding(x)
        
        size = x.shape[-2:]
        # Flatten spatial dimensions: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        x = x.view(-1, self.channels, size[0] * size[1]).swapaxes(1, 2)
        
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        
        # Reshape back: [B, H*W, C] -> [B, C, H*W] -> [B, C, H, W]
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size[0], size[1])
    
# class SelfAttention(nn.Module):
#     # From the original model
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

# class SelfAttention(nn.Module):
#     """Lightweight spatial self-attention (single-head)."""
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
# 
#     def forward(self, x):
#         # x: (B, C, H, W)
#         B, C, H, W = x.shape
#         q = self.query(x).view(B, -1, H * W)                # (B, Cq, N)
#         k = self.key(x).view(B, -1, H * W)                  # (B, Cq, N)
#         v = self.value(x).view(B, -1, H * W)                # (B, C, N)
# 
#         attn = torch.bmm(q.permute(0, 2, 1), k)             # (B, N, N)
#         attn = F.softmax(attn / (q.size(1) ** 0.5), dim=-1) # spatial attention
#         out = torch.bmm(v, attn.permute(0, 2, 1))           # (B, C, N)
#         out = out.view(B, C, H, W)
#         return self.gamma * out + x


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