import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math
from tqdm import tqdm
import os
from typing import Tuple, Optional
import matplotlib.pyplot as plt

from data import EfficientDataset

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timestep encoding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Residual block with time embedding and group normalization"""
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """Self-attention block for better feature representation"""
    def __init__(self, channels, groups=8):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(groups, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.group_norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, c, h * w).transpose(-1, -2), qkv)
        
        attention = F.scaled_dot_product_attention(q, k, v)
        attention = attention.transpose(-1, -2).view(b, c, h, w)
        return x + self.to_out(attention)

class FrameConditionedUNet(nn.Module):
    """U-Net architecture conditioned on previous and next frames"""
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256, 
                 base_channels=64, channel_multipliers=[1, 2, 4, 8]):
        super().__init__()
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Frame conditioning: previous and next frames (2 channels)
        # Plus noisy target frame (1 channel) = 3 total input channels
        self.input_conv = nn.Conv2d(in_channels + 2, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        in_ch = base_channels
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            self.encoder_blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim))
            self.encoder_blocks.append(ResidualBlock(out_ch, out_ch, time_emb_dim))
            
            # Add attention at higher resolutions
            if mult >= 4:
                self.encoder_attentions.append(AttentionBlock(out_ch))
            else:
                self.encoder_attentions.append(nn.Identity())
            
            self.downsample_blocks.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch
        
        # Middle
        mid_ch = base_channels * channel_multipliers[-1]
        self.middle_block1 = ResidualBlock(mid_ch, mid_ch, time_emb_dim)
        self.middle_attention = AttentionBlock(mid_ch)
        self.middle_block2 = ResidualBlock(mid_ch, mid_ch, time_emb_dim)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        channel_multipliers_rev = list(reversed(channel_multipliers))
        for i, mult in enumerate(channel_multipliers_rev):
            out_ch = base_channels * mult
            in_ch = mid_ch if i == 0 else base_channels * channel_multipliers_rev[i-1]
            
            self.upsample_blocks.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
            
            # Skip connection doubles the channels
            self.decoder_blocks.append(ResidualBlock(out_ch * 2, out_ch, time_emb_dim))
            self.decoder_blocks.append(ResidualBlock(out_ch, out_ch, time_emb_dim))
            
            if mult >= 4:
                self.decoder_attentions.append(AttentionBlock(out_ch))
            else:
                self.decoder_attentions.append(nn.Identity())
        
        # Output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, timestep, prev_frame, next_frame):
        # Time embedding
        time_emb = self.time_embedding(timestep)
        time_emb = self.time_mlp(time_emb)
        
        # Concatenate noisy target with conditioning frames
        x_input = torch.cat([x, prev_frame, next_frame], dim=1)
        x = self.input_conv(x_input)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder
        for i in range(len(self.encoder_blocks) // 2):
            x = self.encoder_blocks[i*2](x, time_emb)
            x = self.encoder_blocks[i*2 + 1](x, time_emb)
            x = self.encoder_attentions[i](x)
            skip_connections.append(x)
            x = self.downsample_blocks[i](x)
        
        # Middle
        x = self.middle_block1(x, time_emb)
        x = self.middle_attention(x)
        x = self.middle_block2(x, time_emb)
        
        # Decoder
        for i in range(len(self.decoder_blocks) // 2):
            x = self.upsample_blocks[i](x)
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[i*2](x, time_emb)
            x = self.decoder_blocks[i*2 + 1](x, time_emb)
            x = self.decoder_attentions[i](x)
        
        return self.output_conv(x)

class DDPMScheduler:
    """DDPM noise scheduler"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x_start, noise, timesteps):
        """Forward diffusion process: q(x_t | x_0)"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def denoise_step(self, model, x_t, t, prev_frame, next_frame):
        """Single denoising step"""
        with torch.no_grad():
            # Predict noise
            predicted_noise = model(x_t, t, prev_frame, next_frame)
            
            # Get coefficients
            alpha_t = self.alphas[t][:, None, None, None]
            alpha_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
            beta_t = self.betas[t][:, None, None, None]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            
            # Predict x_0
            x_0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            # Compute x_{t-1}
            x_prev = (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_t)
            
            # Add noise if not the last step
            if t[0] > 0:
                posterior_variance_t = self.posterior_variance[t][:, None, None, None]
                noise = torch.randn_like(x_t)
                x_prev += torch.sqrt(posterior_variance_t) * noise
            
            return x_prev

class OCTFrameInterpolator:
    """Main training and inference class"""
    def __init__(self, model, scheduler, device='cuda'):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device
        
        # Move scheduler tensors to device
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
            attr_value = getattr(self.scheduler, attr_name)
            setattr(self.scheduler, attr_name, attr_value.to(device))

    def train_step(self, batch):
        """Single training step"""
        # batch shape: (B, 3, H, W) - [prev, target, next]
        prev_frame = batch[:, 0:1]  # (B, 1, H, W)
        target_frame = batch[:, 1:2]  # (B, 1, H, W)
        next_frame = batch[:, 2:3]  # (B, 1, H, W)
        
        batch_size = target_frame.shape[0]
        
        # Sample noise and timesteps
        noise = torch.randn_like(target_frame)
        timesteps = self.scheduler.sample_timesteps(batch_size, self.device)
        
        # Add noise to target frame
        noisy_target = self.scheduler.add_noise(target_frame, noise, timesteps)
        
        # Predict noise
        predicted_noise = self.model(noisy_target, timesteps, prev_frame, next_frame)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def generate(self, prev_frame, next_frame, num_inference_steps=50):
        """Generate intermediate frame"""
        self.model.eval()
        batch_size = prev_frame.shape[0]
        
        # Start from pure noise
        x = torch.randn(batch_size, 1, prev_frame.shape[2], prev_frame.shape[3], device=self.device)
        
        # Create inference timesteps
        timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)
        
        for t in tqdm(timesteps, desc="Generating frame"):
            t_batch = t.repeat(batch_size)
            x = self.scheduler.denoise_step(self.model, x, t_batch, prev_frame, next_frame)
        
        return torch.clamp(x, 0, 1)

def train_model(dataset, model, scheduler, num_epochs=100, batch_size=8, lr=1e-4, 
                save_dir='checkpoints', device='cuda'):
    """Training loop"""
    os.makedirs(save_dir, exist_ok=True)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    interpolator = OCTFrameInterpolator(model, scheduler, device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            loss = interpolator.train_step(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        scheduler_lr.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.6f}, LR: {scheduler_lr.get_last_lr()[0]:.2e}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    return interpolator

def visualize_results(interpolator, dataset, num_samples=4, save_path='results.png'):
    """Visualize interpolation results"""
    interpolator.model.eval()
    
    # Get some test samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        sequence = dataset[idx].to(interpolator.device).unsqueeze(0)  # Add batch dim
        prev_frame = sequence[:, 0:1]
        target_frame = sequence[:, 1:2]
        next_frame = sequence[:, 2:3]
        
        # Generate intermediate frame
        with torch.no_grad():
            generated_frame = interpolator.generate(prev_frame, next_frame, num_inference_steps=50)
        
        # Convert to numpy for visualization
        prev_np = prev_frame.cpu().squeeze().numpy()
        target_np = target_frame.cpu().squeeze().numpy()
        next_np = next_frame.cpu().squeeze().numpy()
        generated_np = generated_frame.cpu().squeeze().numpy()
        
        axes[i, 0].imshow(prev_np, cmap='gray')
        axes[i, 0].set_title('Previous Frame')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(target_np, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(generated_np, cmap='gray')
        axes[i, 2].set_title('Generated')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(next_np, cmap='gray')
        axes[i, 3].set_title('Next Frame')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize model and scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = FrameConditionedUNet(
        in_channels=1,
        out_channels=1,
        time_emb_dim=256,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 8]
    )
    
    scheduler = DDPMScheduler(
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02
    )
    
    # Load your dataset
    dataset = EfficientDataset('/home/esad-ugur/Data/OCT/train_all', image_size=128, window_size=3)
    
    # Train the model
    interpolator = train_model(
        dataset=dataset,
        model=model,
        scheduler=scheduler,
        num_epochs=100,
        batch_size=48,
        lr=1e-4,
        device=device
    )
    
    # Visualize results
    visualize_results(interpolator, dataset, num_samples=4)
    
    print("Model architecture created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")