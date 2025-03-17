import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from tqdm import tqdm

# TODO: Try the model

class OctSequenceDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Dataset for OCT sequences
        
        Args:
            data: List of OCT sequences, each with 19 slices
            transform: Optional transform to apply to the data
        """
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        
        # Get odd-numbered slices (1, 3, 5, ..., 19) as condition (10 slices)
        condition = sequence[::2]
        
        # Get even-numbered slices (2, 4, 6, ..., 18) as target (9 slices)
        target = sequence[1::2]
        
        if self.transform:
            condition = self.transform(condition)
            target = self.transform(target)
            
        return condition, target

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb=None):
        residual = self.residual(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        
        if self.time_mlp and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = time_emb[..., None, None]
            x = x + time_emb
            
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        
        return x + residual

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, time_emb_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        
    def forward(self, x, time_emb=None):
        x = self.conv(x, time_emb)
        return self.downsample(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv = ConvBlock(in_channels * 2, out_channels, time_emb_dim)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        
    def forward(self, x, skip_x, time_emb=None):
        x = self.upsample(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x, time_emb)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=(64, 128, 256, 512), time_emb_dim=32, condition_channels=1):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Process condition
        self.condition_encoder = nn.Conv2d(condition_channels, channels[0] // 2, 3, padding=1)
        
        # Initial convolution to process image
        in_ch = in_channels + channels[0] // 2  # Add condition channels to input
        self.init_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)
        
        # Downsampling blocks
        self.downs = nn.ModuleList()
        curr_channels = channels[0]
        
        for next_channels in channels[1:]:
            self.downs.append(DownBlock(curr_channels, next_channels, time_emb_dim))
            curr_channels = next_channels
            
        # Middle block
        self.middle = ConvBlock(channels[-1], channels[-1], time_emb_dim)
        
        # Upsampling blocks
        self.ups = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        
        for i in range(len(channels) - 1):
            self.ups.append(UpBlock(reversed_channels[i], reversed_channels[i+1], time_emb_dim))
            
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1),
            nn.GroupNorm(8, channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1)
        )
        
    def forward(self, x, condition, time):
        # Embed time
        time_emb = self.time_mlp(time)
        
        # Process condition
        encoded_condition = self.condition_encoder(condition)
        
        # Concatenate condition with input
        x = torch.cat([x, encoded_condition], dim=1)
        x = self.init_conv(x)
        
        # Store skip connections
        skip_connections = [x]
        
        # Downsample
        for down in self.downs:
            x = down(x, time_emb)
            skip_connections.append(x)
            
        # Middle
        x = self.middle(x, time_emb)
        
        # Upsample with skip connections
        for up, skip in zip(self.ups, reversed(skip_connections[:-1])):
            x = up(x, skip, time_emb)
            
        # Final convolution
        return self.final_conv(x)

class GaussianDiffusion:
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, timesteps=1000, device="cuda"):
        """
        Gaussian Diffusion process for OCT slice generation
        
        Args:
            model: U-Net model to predict noise
            beta_start: Start value for noise schedule
            beta_end: End value for noise schedule
            timesteps: Number of diffusion steps
            device: Device to run on
        """
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        
        # Pre-compute diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to the input according to the noise schedule
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_0, condition, t, noise=None):
        """
        Training loss for the denoising model
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Get noisy samples
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict the noise
        predicted_noise = self.model(x_t, condition, t)
        
        # Calculate loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, x_t, condition, t):
        """
        Single step of the reverse diffusion process: p(x_{t-1} | x_t)
        """
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.rsqrt(self.alphas[t]).reshape(-1, 1, 1, 1)
        
        # Equation 11 in the DDPM paper
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * self.model(x_t, condition, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, condition, shape):
        """
        Sample from the model
        """
        # Start from pure noise
        img = torch.randn(shape, device=self.device)
        
        # Iteratively denoise
        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            img = self.p_sample(img, condition, t_batch)
            
        return img

def train_diffusion_model(diffusion, dataloader, epochs, lr=1e-4, device="cuda"):
    """
    Train the diffusion model
    """
    optimizer = optim.AdamW(diffusion.model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        diffusion.model.train()
        total_loss = 0
        
        for step, (condition, target) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            condition = condition.to(device)
            target = target.to(device)
            
            # Randomly sample timesteps
            batch_size = condition.shape[0]
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            
            # Calculate loss
            loss = diffusion.p_losses(target, condition, t)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
# Example usage
def main():
    # Hyperparameters
    batch_size = 16
    epochs = 50
    timesteps = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Assuming your dataset is already loaded as a list of OCT sequences
    # Each sequence is a tensor of shape [19, height, width] representing 19 slices
    # Create a placeholder for your actual data loading
    data = []  # Replace with your actual data loading
    
    # Create dataset and dataloader
    dataset = OctSequenceDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Get shapes from a sample
    sample_condition, sample_target = next(iter(dataloader))
    input_channels = sample_target.shape[1]  # Usually 1 for grayscale OCT images
    condition_channels = sample_condition.shape[1]  # Usually 1 for grayscale OCT images
    
    # Create model, assuming OCT slices are grayscale
    model = UNet(in_channels=1, out_channels=1, channels=(64, 128, 256, 512), 
                 time_emb_dim=32, condition_channels=1).to(device)
    
    # Create diffusion process
    diffusion = GaussianDiffusion(model, timesteps=timesteps, device=device)
    
    # Train the model
    train_diffusion_model(diffusion, dataloader, epochs=epochs, device=device)
    
    # Save the trained model
    torch.save(model.state_dict(), "oct_diffusion_model.pt")
    
    # Generate samples
    model.eval()
    sample_condition = sample_condition.to(device)
    
    # Generate 9 even-numbered slices
    generated_samples = diffusion.sample(sample_condition, 
                                         shape=(sample_condition.shape[0], 1, 
                                                sample_condition.shape[2], 
                                                sample_condition.shape[3]))
    
    print("Generated samples shape:", generated_samples.shape)
    
if __name__ == "__main__":
    main()