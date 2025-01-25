import numpy as np
import torch

def reconstruct_image(magnitude, phase):
    """Reconstruct image from magnitude and phase"""
    # Denormalize magnitude and phase
    magnitude = (magnitude + 1) / 2  # Back to [0, 1]
    magnitude = np.expm1(magnitude * (magnitude.max() - magnitude.min()) + magnitude.min())
    
    phase = phase * np.pi  # Back to [-π, π]
    
    # Combine magnitude and phase
    complex_spectrum = magnitude * np.exp(1j * phase)
    
    # Inverse FFT
    inverse_shift = np.fft.ifftshift(complex_spectrum)
    image = np.fft.ifft2(inverse_shift)
    image = np.abs(image)
    
    # Normalize output image
    image = (image - image.min()) / (image.max() - image.min())
    
    return image

@torch.no_grad()
def sample_diffusion(model, condition, device, shape):
    # Sample from the diffusion model
    x = torch.randn(shape).to(device)
    
    for t in reversed(range(model.timesteps)):
        timesteps = torch.full((shape[0],), t, device=device, dtype=torch.long)
        predicted_noise = model(x, condition, timesteps)
        
        alpha = model.alpha[t]
        alpha_bar = model.alpha_bar[t]
        beta = model.beta[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + \
            torch.sqrt(beta) * noise
    # print(x.shape)
    return x


