import torch
import torch.nn as nn
import numpy as np

# TODO: Implement a GAN model

class ComplexUNet(nn.Module):
    def __init__(self, input_channels, condition_channels, hidden_channels, time_embed_dim):
        super().__init__()
        
        # Double the channels to handle both magnitude and phase
        self.input_channels = input_channels * 2
        self.condition_channels = condition_channels * 2
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # U-Net architecture
        self.inc = self._double_conv(self.input_channels + self.condition_channels + time_embed_dim, hidden_channels)
        self.down1 = self._down_block(hidden_channels, hidden_channels * 2)
        self.down2 = self._down_block(hidden_channels * 2, hidden_channels * 4)
        
        self.up1 = self._up_block(hidden_channels * 4, hidden_channels * 2)
        self.up2 = self._up_block(hidden_channels * 2, hidden_channels)
        
        self.outc = nn.Conv2d(hidden_channels, self.input_channels, kernel_size=1)

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

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self._double_conv(out_channels, out_channels)
        )

    def forward(self, x, condition, t):
        # x shape: (B, T*2, H, W) - contains magnitude and phase
        t = self.time_mlp(t.float().view(-1, 1))
        t = t.view(-1, t.shape[-1], 1, 1).expand(-1, -1, x.shape[-2], x.shape[-1])
        
        x = torch.cat([x, condition, t], dim=1)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        x = self.up2(x)
        x = self.outc(x)
        
        return x

class ComplexUNetLarge(nn.Module):
    def __init__(self, input_channels, condition_channels, hidden_channels, time_embed_dim):
        super().__init__()
        
        # Double the channels to handle both magnitude and phase
        self.input_channels = input_channels * 2
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
        self.outc = nn.Conv2d(hidden_channels, self.input_channels, kernel_size=1)
    
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
    
    def forward(self, x, condition, t):
        # Time embedding
        t = self.time_mlp(t.float().view(-1, 1))
        t = t.view(-1, t.shape[-1], 1, 1).expand(-1, -1, x.shape[-2], x.shape[-1])
        
        # Initial concatenation
        x = torch.cat([x, condition, t], dim=1)
        
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

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 8, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0] * size[1]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size[0], size[1])

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
        frame1 = frame1.reshape((frame1.shape[0], 1, frame1.shape[1], frame1.shape[2]))
        frame2 = frame2.reshape((frame2.shape[0], 1, frame2.shape[1], frame2.shape[2]))
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


class DiffusionInterpolator(nn.Module):
    def __init__(self, input_channels, hidden_channels, device='cuda', timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.input_channels = input_channels * 2  # *2 for complex input
        
        # Beta schedule
        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)
        
        # Time embedding
        time_embed_dim = hidden_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_channels),
            nn.Linear(hidden_channels, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # U-Net
        self.inc = self._double_conv(self.input_channels * 3 + time_embed_dim, hidden_channels)  # *2 for condition
        self.down1 = self._down_block(hidden_channels, hidden_channels * 2)
        self.down2 = self._down_block(hidden_channels * 2, hidden_channels * 4)
        self.down3 = self._down_block(hidden_channels * 4, hidden_channels * 8)
        
        self.bridge = nn.Sequential(
            self._double_conv(hidden_channels * 8, hidden_channels * 8),
            SelfAttention(hidden_channels * 8)
        )
        
        self.up3 = nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, kernel_size=2, stride=2)
        self.conv_up3 = self._double_conv(hidden_channels * 8, hidden_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=2, stride=2)
        self.conv_up2 = self._double_conv(hidden_channels * 4, hidden_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2)
        self.conv_up1 = self._double_conv(hidden_channels * 2, hidden_channels)
        
        self.outc = nn.Conv2d(hidden_channels, self.input_channels, kernel_size=1)
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
    
    def _down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self._double_conv(in_channels, out_channels)
        )
    
    def get_noisy_image(self, x, t):
        # Add noise to image according to diffusion schedule
        a = self.alpha_bar[t][:, None, None, None]
        noise = torch.randn_like(x)
        return torch.sqrt(a) * x + torch.sqrt(1 - a) * noise, noise
    
    def forward(self, x, condition, t):
        # Time embedding
        t = self.time_mlp(t)
        t = t.view(-1, t.shape[-1], 1, 1).expand(-1, -1, x.shape[-2], x.shape[-1])
        
        # Concatenate input and condition
        x = torch.cat([x, condition, t], dim=1)
        
        # U-Net forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x4 = self.bridge(x4)
        
        x = self.up3(x4)
        x = self.conv_up3(torch.cat([x, x3], dim=1))
        
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x2], dim=1))
        
        x = self.up1(x)
        x = self.conv_up1(torch.cat([x, x1], dim=1))
        
        return self.outc(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

