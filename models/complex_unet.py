import torch
import torch.nn as nn
import numpy as np

from .generic import SelfAttention

class ComplexUNetLarge(nn.Module):
    def __init__(self, input_channels, condition_channels, hidden_channels, time_embed_dim):
        super().__init__()
        
        # Double the channels to handle both magnitude and phase
        # self.input_channels = input_channels * 2
        self.input_channels = 0
        self.condition_channels = condition_channels * 2
        
        # Enhanced time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )
        
        # Calculate initial channels after concatenating input, condition, and time embedding
        initial_channels = self.input_channels + self.condition_channels + time_embed_dim
        
        # Encoder path
        self.inc = self._double_conv(initial_channels, hidden_channels)
        self.down1 = self._down_block(hidden_channels, hidden_channels * 2)
        self.down2 = self._down_block(hidden_channels * 2, hidden_channels * 4)
        self.down3 = self._down_block(hidden_channels * 4, hidden_channels * 8)
        self.down4 = self._down_block(hidden_channels * 8, hidden_channels * 8)  # Limit maximum channels
        
        # Bridge
        self.bridge = nn.Sequential(
            self._double_conv(hidden_channels * 8, hidden_channels * 8),
            SelfAttention(hidden_channels * 8)
        )
        
        # Decoder path
        self.up4 = nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 8, kernel_size=2, stride=2)
        self.conv_up4 = self._double_conv(hidden_channels * 16, hidden_channels * 8)  # 16 due to skip connection
        
        self.up3 = nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, kernel_size=2, stride=2)
        self.conv_up3 = self._double_conv(hidden_channels * 8, hidden_channels * 4)  # 8 due to skip connection
        
        self.up2 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=2, stride=2)
        self.conv_up2 = self._double_conv(hidden_channels * 4, hidden_channels * 2)  # 4 due to skip connection
        
        self.up1 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2)
        self.conv_up1 = self._double_conv(hidden_channels * 2, hidden_channels)  # 2 due to skip connection
        
        # Output layer
        self.outc = nn.Conv2d(hidden_channels, 2, kernel_size=1)
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def _down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self._double_conv(in_channels, out_channels)
        )
    
    def forward(self, condition, t):
        # Time embedding
        t = self.time_mlp(t.float().view(-1, 1))
        t = t.view(-1, t.shape[-1], 1, 1).expand(-1, -1, condition.shape[-2], condition.shape[-1])
        
        # Initial concatenation
        # x = torch.cat([x, condition, t], dim=1)
        x = torch.cat([condition, t], dim=1)

        # Encoder path with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Bridge
        x5 = self.bridge(x5)
        
        # Decoder path with skip connections
        x = self.up4(x5)
        x = self.conv_up4(torch.cat([x, x4], dim=1))
        
        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, x3], dim=1))
        
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x2], dim=1))
        
        x = self.up1(x)
        x = self.conv_up1(torch.cat([x, x1], dim=1))
        
        # Output projection
        x = self.outc(x)
        return x