import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm=None, activation=None):
        super(ConvBlock, self).__init__()
        # padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Normalization
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = None
            
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = None
            
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, 3, stride=1, padding=1, norm='instance', activation='relu')
        self.conv2 = ConvBlock(channels, channels, 3, stride=1, padding=1, norm='instance', activation=None)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual  # Skip connection
        return self.relu(out)

class Encoder(nn.Module):
    def __init__(self, in_channels, initial_filters=64):
        super(Encoder, self).__init__()
        
        # Initial feature extraction
        self.initial_conv = ConvBlock(in_channels, initial_filters, 7, stride=1, padding= 3, norm='instance', activation='relu')
        
        # Downsampling path
        self.down1 = ConvBlock(initial_filters, initial_filters*2, 3, stride=1, padding=1, norm='instance', activation='relu')
        self.down2 = ConvBlock(initial_filters*2, initial_filters*2, 3, stride=2, padding=1, norm=None, activation='relu')
        self.down3 = ConvBlock(initial_filters*2, initial_filters*4, 3, stride=1, padding=1, norm='instance', activation='relu')
        self.down4 = ConvBlock(initial_filters*4, initial_filters*4, 3, stride=2, padding=1, norm=None, activation='relu')
        
        # ResNet blocks for feature processing
        self.resblocks = nn.Sequential(
            ResNetBlock(initial_filters*4),
            ResNetBlock(initial_filters*4),
            ResNetBlock(initial_filters*4),
            ResNetBlock(initial_filters*4),
            ResNetBlock(initial_filters*4)
        )
        
    def forward(self, x):
        # Initial layers with 2x features as shown in diagram
        x1 = self.initial_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        features = self.resblocks(x5)
        
        return features, x1  # Return both final features and skip connection from first layer

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.resblocks = nn.Sequential(
            ResNetBlock(in_channels),
            ConvBlock(in_channels, in_channels, 1, stride=1, padding=0, norm=None, activation='relu'),
            ResNetBlock(in_channels),
            ConvBlock(in_channels, in_channels, 1, stride=1, padding=0, norm=None, activation='relu'),
            ResNetBlock(in_channels),
            ResNetBlock(in_channels),
        )
        
    def forward(self, x):
        return self.resblocks(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, final_channels=1):
        super(Decoder, self).__init__()
        
        # UpSampling path
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv1 = ConvBlock(in_channels, in_channels//2, 3, stride=1, padding=1, norm='instance', activation='relu')
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv2 = ConvBlock(in_channels//2, in_channels//4, 3, stride=1, padding=1, norm='instance', activation='relu')
        
        # Final 1x1 convolution to produce the output image
        self.final_conv = ConvBlock(in_channels//4, final_channels, 7, stride=1, padding=3, norm=None, activation='tanh')
        
    def forward(self, x, skip_connection=None):
        # First upsampling
        x = self.up1(x)
        x = self.up_conv1(x)
        
        # Second upsampling
        x = self.up2(x)
        x = self.up_conv2(x)
        
        # Optional skip connection from encoder
        if skip_connection is not None:
            x = x + skip_connection
            
        # Final convolution
        x = self.final_conv(x)
        
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, initial_filters=64):
        super(Generator, self).__init__()
        
        # Components
        self.encoder = Encoder(in_channels, initial_filters)
        self.bottleneck = Bottleneck(initial_filters*8, initial_filters*4)
        self.decoder = Decoder(initial_filters*8, out_channels)
        
    def forward(self, pre, post):
        # Encoder path
        features_pre, _ = self.encoder(pre)
        features_post, _ = self.encoder(post)

        features = torch.concat((features_pre, features_post), dim=1)
        # Bottleneck
        bottleneck_features = self.bottleneck(features)
        # Decoder path
        output = self.decoder(bottleneck_features)
        
        return output

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, hidden_features=32):
        super(Discriminator, self).__init__()

        self.down1 = nn.Sequential(
            ConvBlock(in_channels, in_channels, 4, stride=1, padding=1, norm='instance', activation='leaky'),
            ConvBlock(hidden_features, hidden_features*2, 3, stride=2, padding=1, norm=None, activation='leaky')
        )

        self.down2 = nn.Sequential(
            ConvBlock(hidden_features*2, hidden_features*2, 4, stride=1, padding=1, norm='instance', activation='leaky'),
            ConvBlock(hidden_features*2, hidden_features*4, 3, stride=2, padding=1, norm=None, activation='leaky')
        )

        self.down3 = nn.Sequential(
            ConvBlock(hidden_features*4, hidden_features*4, 4, stride=1, padding=1, norm='instance', activation='leaky'),
            ConvBlock(hidden_features*4, hidden_features*8, 3, stride=2, padding=1, norm=None, activation='leaky'),
            ConvBlock(hidden_features*8, hidden_features*8, 4, stride=1, padding=1, norm='instance', activation='leaky'),
            ConvBlock(hidden_features*8, hidden_features*16, 4, stride=1, padding=1, norm=None, activation='leaky'),
        )
    
    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        return x

# Example usage
def test_model():
    # Create a sample input (batch_size=1, channels=1, height=256, width=256)
    # This simulates one grayscale medical image
    pre = torch.randn(1, 1, 256, 256)
    post = torch.randn(1, 1, 256, 256)
    
    # Initialize the model
    model = Generator(in_channels=1, out_channels=1, initial_filters=64)
    model_d = Discriminator()
    
    # Forward pass
    output = model(pre, post)

    dis_out = model_d(output)

    g_params = sum(p.numel() for p in model.parameters())
    d_params = sum(p.numel() for p in model_d.parameters())
    
    print(f"Input shape: {pre.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Discriminator output shape: {dis_out.shape}")
    print(f"Total parameters: {g_params + d_params}")
    
    return model

if __name__ == "__main__":
    model = test_model()