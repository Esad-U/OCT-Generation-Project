import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .generic import SelfAttention, SEBlock, ResidualBlock

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


class UNetUpsampleEnhanced(nn.Module):
    def __init__(self, hidden_channels=64, in_channels=2, out_channels=1, dropout=0.3):
        """
        in_channels: 2 (two grayscale frames concatenated)
        out_channels: 1 (interpolated grayscale frame)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        hc = hidden_channels

        # Encoder
        self.inc = ResidualBlock(in_channels, hc)                # x1: H
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(hc, hc * 2))   # x2: H/2
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(hc * 2, hc * 4)) # x3: H/4
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(hc * 4, hc * 8)) # x4: H/8

        # Bridge
        self.bridge = nn.Sequential(
            ResidualBlock(hc * 8, hc * 8),
            nn.Dropout(p=dropout),
            SelfAttention(hc * 8)
        )

        # Decoder (bilinear upsample + residual conv)
        # up3: H/8 -> H/4, concat with x3 (hc*4) => in_channels = hc*8 + hc*4
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = ResidualBlock(hc * 8 + hc * 4, hc * 4)
        self.se3 = SEBlock(hc * 4)

        # up2: H/4 -> H/2, concat with x2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = ResidualBlock(hc * 4 + hc * 2, hc * 2)
        self.se2 = SEBlock(hc * 2)

        # up1: H/2 -> H, concat with x1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = ResidualBlock(hc * 2 + hc, hc)

        # Final 1x1 conv to output channels
        self.outc = nn.Conv2d(hc, out_channels, kernel_size=1)

        # Deep supervision heads (1x1 convs)
        self.ds3 = nn.Conv2d(hc * 4, out_channels, kernel_size=1)  # from conv_up3 (H/4 -> upsample to H)
        self.ds2 = nn.Conv2d(hc * 2, out_channels, kernel_size=1)  # from conv_up2 (H/2 -> upsample to H)

        # Activation
        self.final_activation = nn.Tanh()  # assume input images normalized to [0,1]

    def forward(self, frame1, frame2):
        """
        Returns:
            out: (B, 1, H, W) final output
            ds_outputs: dict with 'ds2' and 'ds3' upsampled to full resolution (use for aux losses)
        """
        x = torch.cat([frame1, frame2], dim=1)  # (B, 2, H, W)

        # Encoder
        x1 = self.inc(x)     # H
        x2 = self.down1(x1)  # H/2
        x3 = self.down2(x2)  # H/4
        x4 = self.down3(x3)  # H/8

        # Bridge
        b = self.bridge(x4)  # H/8

        # Decoder
        u3 = self.up3(b)                              # H/4
        u3 = torch.cat([u3, x3], dim=1)               # concat along channels
        u3 = self.conv_up3(u3)                        # H/4
        u3 = self.se3(u3)

        u2 = self.up2(u3)                             # H/2
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)                        # H/2
        u2 = self.se2(u2)

        u1 = self.up1(u2)                             # H
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.conv_up1(u1)                        # H

        # Final output
        out = self.outc(u1)                           # (B, out_channels, H, W)
        out = self.final_activation(out)

        # Deep supervision outputs (upsample to full resolution)
        ds3 = self.ds3(u3)  # H/4
        ds3_up = F.interpolate(ds3, size=out.shape[2:], mode='bilinear', align_corners=True)

        ds2 = self.ds2(u2)  # H/2
        ds2_up = F.interpolate(ds2, size=out.shape[2:], mode='bilinear', align_corners=True)

        ds_outputs = {'ds2': ds2_up, 'ds3': ds3_up}

        return {
            "main": out,   # final full-resolution prediction
            "ds2": ds2_up, # deep supervision output from stage 2
            "ds3": ds3_up  # deep supervision output from stage 3
        }