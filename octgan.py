import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging


class Generator(nn.Module):
    def __init__(self, input_channels=1, base_filters=128):
        super(Generator, self).__init__()
        
        # Larger encoder for first input image
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_filters, base_filters*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_filters*2, base_filters*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_filters*4, base_filters*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_filters*8, base_filters*16, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*16),
            nn.LeakyReLU(0.2)
        )
        
        # Larger encoder for second input image
        self.encoder2 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_filters, base_filters*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_filters*2, base_filters*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_filters*4, base_filters*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_filters*8, base_filters*16, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*16),
            nn.LeakyReLU(0.2)
        )
        
        # Enhanced attention mechanism with more heads
        self.attention1 = nn.MultiheadAttention(base_filters*16, 16)
        self.attention2 = nn.MultiheadAttention(base_filters*16, 16)
        
        # Additional processing layers after attention
        self.post_attention = nn.Sequential(
            nn.Conv2d(base_filters*32, base_filters*16, 3, padding=1),
            nn.BatchNorm2d(base_filters*16),
            nn.LeakyReLU(0.2)
        )
        
        # Larger decoder with skip connections
        self.decoder = nn.ModuleList([
            # Layer 1
            nn.Sequential(
                nn.ConvTranspose2d(base_filters*16, base_filters*16, 4, stride=2, padding=1),
                nn.BatchNorm2d(base_filters*16),
                nn.ReLU()
            ),
            # Layer 2
            nn.Sequential(
                nn.ConvTranspose2d(base_filters*16*2, base_filters*8, 4, stride=2, padding=1),
                nn.BatchNorm2d(base_filters*8),
                nn.ReLU()
            ),
            # Layer 3
            nn.Sequential(
                nn.ConvTranspose2d(base_filters*8*2, base_filters*4, 4, stride=2, padding=1),
                nn.BatchNorm2d(base_filters*4),
                nn.ReLU()
            ),
            # Layer 4
            nn.Sequential(
                nn.ConvTranspose2d(base_filters*4*2, base_filters*2, 4, stride=2, padding=1),
                nn.BatchNorm2d(base_filters*2),
                nn.ReLU()
            ),
            # Final layer
            nn.Sequential(
                nn.ConvTranspose2d(base_filters*2*2, input_channels, 4, stride=2, padding=1),
                nn.Tanh()
            )
        ])
        
        # Refinement network
        self.refinement = nn.Sequential(
            nn.Conv2d(input_channels*2, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x1, x2):
        # Store encoder features for skip connections
        enc1_features = []
        enc2_features = []
        
        # First encoder
        x1_curr = x1
        for layer in self.encoder1:
            x1_curr = layer(x1_curr)
            enc1_features.append(x1_curr)
        
        # print(len(enc1_features))
        # for i in enc1_features:
        #     print(f"enc1:{i.shape}")
        
        # Second encoder
        x2_curr = x2
        for layer in self.encoder2:
            x2_curr = layer(x2_curr)
            enc2_features.append(x2_curr)
        
        # Multi-head attention
        b, c, h, w = x1_curr.shape
        feat1_flat = x1_curr.flatten(2).permute(2, 0, 1)
        feat2_flat = x2_curr.flatten(2).permute(2, 0, 1)
        
        attn_output1, _ = self.attention1(feat1_flat, feat2_flat, feat2_flat)
        attn_output2, _ = self.attention2(feat2_flat, feat1_flat, feat1_flat)
        
        attn_output1 = attn_output1.permute(1, 2, 0).view(b, c, h, w)
        attn_output2 = attn_output2.permute(1, 2, 0).view(b, c, h, w)

        # Combine attention outputs
        combined = torch.cat([attn_output1, attn_output2], dim=1)
        combined = self.post_attention(combined)

        # Decoder with skip connections
        x = combined
        for i, decoder_layer in enumerate(self.decoder[:-1]):
            x = decoder_layer(x)
            # Add skip connections from both encoders
            x = torch.cat([x, enc1_features[-3*i-4], enc2_features[-3*i-4]], dim=1)
        
        # Final decoder layer
        x = self.decoder[-1](x)
        
        # Refinement
        x = torch.cat([x, x1], dim=1)  # Concatenate with input for refinement
        x = self.refinement(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, base_filters=128):
        super(Discriminator, self).__init__()
        
        self.features = nn.Sequential(
            # Initial layer
            nn.Conv2d(input_channels, base_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # Deeper feature extraction
            nn.Conv2d(base_filters, base_filters*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_filters*2, base_filters*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_filters*4, base_filters*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_filters*8, base_filters*16, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*16),
            nn.LeakyReLU(0.2),
            
            # Final layers
            nn.Conv2d(base_filters*16, base_filters*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_filters*8, 1, 2, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        # Additional dense layers for global features
        self.global_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, base_filters*8, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_filters*8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.features(x)
        global_features = self.global_features(features)
        out = features + global_features
        return out.clamp(0, 1)


class OCTGAN:
    def __init__(self, device='cuda', base_filters=128):
        self.device = device
        self.generator = Generator(base_filters=base_filters).to(device)
        self.discriminator = Discriminator(base_filters=base_filters).to(device)
        
        # Use larger learning rates and different beta values
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=0.0004, 
            betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=0.0001, 
            betas=(0.5, 0.999)
        )
        
        # Multiple loss functions for better training
        self.adversarial_loss = nn.BCELoss()
        self.reconstruction_loss = nn.L1Loss()
        self.perceptual_loss = nn.MSELoss()  # Can be replaced with a VGG-based loss
        
    def train_step(self, odd_frames, even_frames):
        batch_size = odd_frames.size(0)
        sequence_length = even_frames.size(1)
        
        total_g_loss = 0
        total_d_loss = 0
        
        for i in range(sequence_length):
            img1 = odd_frames[:, i].to(self.device)
            img2 = odd_frames[:, i + 1].to(self.device)
            target = even_frames[:, i].to(self.device)

            b, h, w = img1.shape
            img1 = img1.view(b, 1, h, w)
            img2 = img2.view(b, 1, h, w)
            target = target.view(b, 1, h, w)
            
            real_label = torch.ones(batch_size, 1, 1, 1).to(self.device)
            fake_label = torch.zeros(batch_size, 1, 1, 1).to(self.device)
            
            # Train Generator with multiple loss components
            self.g_optimizer.zero_grad()
            
            generated = self.generator(img1, img2)
            
            fake_concat = torch.cat([img1, generated, img2], dim=1)
            g_loss_adv = self.adversarial_loss(self.discriminator(fake_concat), real_label)
            
            g_loss_rec = self.reconstruction_loss(generated, target)
            g_loss_perceptual = self.perceptual_loss(generated, target)
            
            g_loss = g_loss_adv + 100 * g_loss_rec + 10 * g_loss_perceptual
            g_loss.backward()
            self.g_optimizer.step()
            
            # Train Discriminator with real and fake samples
            self.d_optimizer.zero_grad()
            
            real_concat = torch.cat([img1, target, img2], dim=1)
            real_loss = self.adversarial_loss(self.discriminator(real_concat), real_label)
            
            fake_concat = torch.cat([img1, generated.detach(), img2], dim=1)
            fake_loss = self.adversarial_loss(self.discriminator(fake_concat), fake_label)
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            self.d_optimizer.step()
            
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
        
        return {
            'g_loss': total_g_loss / sequence_length,
            'd_loss': total_d_loss / sequence_length,
        }
    
    def train(self, train_loader, num_epochs, checkpoint_dir, checkpoint_freq, log_interval=10):
        os.makedirs(checkpoint_dir, exist_ok=True)
        for epoch in range(num_epochs):
            epoch_losses = []
            for batch_idx, (odd_frames, even_frames) in enumerate(train_loader):
                losses = self.train_step(odd_frames, even_frames)
                epoch_losses.append(losses)
                
                if batch_idx % log_interval == 0:
                    logging.info(f"Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], "
                          f"G_Loss: {losses['g_loss']:.4f}, D_Loss: {losses['d_loss']:.4f}")
            
            avg_losses = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) 
                         for k in epoch_losses[0].keys()}
            logging.info(f"Epoch [{epoch}/{num_epochs}] Averages: ", avg_losses)

            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': self.g_optimizer.state_dict(),
                    'train_loss': avg_losses,
                    'train_losses': epoch_losses,
                }, checkpoint_path)
                logging.info(f'Saved checkpoint to {checkpoint_path}')
        
        return epoch_losses

    def generate_sequence(self, odd_frames):
        self.generator.eval()
        with torch.no_grad():
            generated_frames = []
            for i in range(odd_frames.size(1) - 1):
                img1 = odd_frames[:, i].to(self.device)
                img2 = odd_frames[:, i + 1].to(self.device)
                generated = self.generator(img1, img2)
                generated_frames.append(generated)
        self.generator.train()
        return torch.stack(generated_frames, dim=1)
