import torch
import torch.nn as nn
import numpy as np

from .generic import SelfAttention

class InterpolationUNet(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        
        # Input is two surrounding frames concatenated
        self.input_channels = input_channels * 2  # *2 for two frames, *2 for complex input
        
        # Encoder
        self.inc = self._double_conv(self.input_channels, hidden_channels)
        self.down1 = self._down_block(hidden_channels, hidden_channels * 2)
        self.down2 = self._down_block(hidden_channels * 2, hidden_channels * 4)
        self.down3 = self._down_block(hidden_channels * 4, hidden_channels * 8)
        
        # Bridge with attention
        self.bridge = nn.Sequential(
            self._double_conv(hidden_channels * 8, hidden_channels * 8),
            SelfAttention(hidden_channels * 8)
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, kernel_size=2, stride=2)
        self.conv_up3 = self._double_conv(hidden_channels * 8, hidden_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=2, stride=2)
        self.conv_up2 = self._double_conv(hidden_channels * 4, hidden_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2)
        self.conv_up1 = self._double_conv(hidden_channels * 2, hidden_channels)
        
        # Output layer (2 channels for complex output)
        self.outc = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
    
    def _down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self._double_conv(in_channels, out_channels)
        )
    
    def forward(self, frame1, frame2):
        # Concatenate input frames
        x = torch.cat([frame1, frame2], dim=1)
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bridge
        x4 = self.bridge(x4)
        
        # Decoder with skip connections
        x = self.up3(x4)
        x = self.conv_up3(torch.cat([x, x3], dim=1))
        
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x2], dim=1))
        
        x = self.up1(x)
        x = self.conv_up1(torch.cat([x, x1], dim=1))
        
        return self.outc(x).squeeze(1)

class UNetUpsample(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        
        # Input is two surrounding frames concatenated
        self.input_channels = input_channels * 2  # *2 for two frames, *2 for complex input
        
        # Encoder
        self.inc = self._double_conv(self.input_channels, hidden_channels)
        self.down1 = self._down_block(hidden_channels, hidden_channels * 2)
        self.down2 = self._down_block(hidden_channels * 2, hidden_channels * 4)
        self.down3 = self._down_block(hidden_channels * 4, hidden_channels * 8)
        
        # Bridge with attention
        self.bridge = nn.Sequential(
            self._double_conv(hidden_channels * 8, hidden_channels * 8),
            SelfAttention(hidden_channels * 8)
        )
        
        # Decoder with upsampling instead of transposed convolutions
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = self._double_conv(hidden_channels * 8 + hidden_channels * 4, hidden_channels * 4)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = self._double_conv(hidden_channels * 4 + hidden_channels * 2, hidden_channels * 2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = self._double_conv(hidden_channels * 2 + hidden_channels, hidden_channels)
        
        # Output layer (2 channels for complex output)
        self.outc = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self._double_conv(in_channels, out_channels)
        )
    
    def forward(self, frame1, frame2):
        # Concatenate input frames
        x = torch.cat([frame1, frame2], dim=1)
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bridge
        x4 = self.bridge(x4)
        
        # Decoder with skip connections
        x = self.up3(x4)
        x = self.conv_up3(torch.cat([x, x3], dim=1))
        
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x2], dim=1))
        
        x = self.up1(x)
        x = self.conv_up1(torch.cat([x, x1], dim=1))
        
        return nn.Tanh()(self.outc(x).squeeze(1))