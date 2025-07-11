import torch
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from skimage.exposure import match_histograms

from utils import reconstruct_image, sample_diffusion

def visualize_results(original_odd, original_even, generated_even, save_path=None):
    """Visualize original and generated sequences with reconstructed images"""
    num_timesteps = original_even.shape[0]
    fig, axes = plt.subplots(6, num_timesteps, figsize=(20, 16))
    
    # Plot titles
    axes[0, num_timesteps//2].set_title("Original Odd Magnitudes", pad=10)
    axes[1, num_timesteps//2].set_title("Original Odd Phases", pad=10)
    axes[2, num_timesteps//2].set_title("Original Even Magnitudes", pad=10)
    axes[3, num_timesteps//2].set_title("Original Even Phases", pad=10)
    axes[4, num_timesteps//2].set_title("Generated Even Magnitudes", pad=10)
    axes[5, num_timesteps//2].set_title("Generated Even Phases", pad=10)
    
    # Helper function to plot magnitude/phase pairs
    def plot_fourier_pair(mag, phase, ax1, ax2):
        im1 = ax1.imshow(mag, cmap='viridis')
        im2 = ax2.imshow(phase, cmap='twilight')
        ax1.axis('off')
        ax2.axis('off')
        return im1, im2
    
    # Plot all sequences
    for t in range(num_timesteps):
        if t < len(original_odd):
            plot_fourier_pair(original_odd[t, 0], original_odd[t, 1], axes[0, t], axes[1, t])
        plot_fourier_pair(original_even[t, 0], original_even[t, 1], axes[2, t], axes[3, t])
        plot_fourier_pair(generated_even[t, 0], generated_even[t, 1], axes[4, t], axes[5, t])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")
    
    # plt.show()

def visualize_reconstructions(original_odd, original_even, generated_even, save_path=None):
    """Visualize reconstructed images"""
    num_timesteps = original_even.shape[0]
    fig, axes = plt.subplots(3, num_timesteps, figsize=(20, 8))
    
    # Plot titles
    axes[0, num_timesteps//2].set_title("Reconstructed Odd Frames", pad=10)
    axes[1, num_timesteps//2].set_title("Original Even Frames", pad=10)
    axes[2, num_timesteps//2].set_title("Generated Even Frames", pad=10)
    
    # Reconstruct and plot images
    for t in range(num_timesteps):
        if t < len(original_odd):
            recon_odd = reconstruct_image(original_odd[t, 0], original_odd[t, 1])
            axes[0, t].imshow(recon_odd, cmap='gray')
            axes[0, t].axis('off')
            
        recon_orig_even = reconstruct_image(original_even[t, 0], original_even[t, 1])
        recon_gen_even = reconstruct_image(generated_even[t, 0], generated_even[t, 1])
        # A trick to reconstruct the even frames using the odd frames
        # recon_gen_even = reconstruct_image((original_odd[t, 0] + original_odd[t+1, 0]) / 2, (original_odd[t, 1] + original_odd[t+1, 1]) / 2)
        
        axes[1, t].imshow(recon_orig_even, cmap='gray')
        axes[2, t].imshow(recon_gen_even, cmap='gray')
        axes[1, t].axis('off')
        axes[2, t].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved reconstructions to {save_path}")
    
    # plt.show()

def visualize_interpolations(original_odd, original_even, generated_even, eval_metrics, save_path=None):
    """Visualize reconstructed images"""
    num_timesteps = original_even.shape[0]
    fig, axes = plt.subplots(3, num_timesteps, figsize=(20, 8))
    
    # Plot titles
    axes[0, num_timesteps//2].set_title("Odd Frames", pad=10)
    axes[1, num_timesteps//2].set_title("Original Even Frames", pad=10)
    axes[2, num_timesteps//2].set_title("Generated Even Frames", pad=10)
    # add text below the figure
    fig.text(0.5, 0.04, f'SSIM: {eval_metrics['ssim']}, PSNR: {eval_metrics['psnr']}, FID: {eval_metrics['fid']}, LPIPS: {eval_metrics['lpips']}', ha='center', va='center', fontsize=14)

    # Reconstruct and plot images
    for t in range(num_timesteps):
        if t < len(original_odd):
            axes[0, t].imshow(original_odd[t], cmap='gray')
            axes[0, t].axis('off')
        # A trick to reconstruct the even frames using the odd frames
        # recon_gen_even = reconstruct_image(generated_even[t, 0], (original_odd[t, 1] + original_odd[t+1, 1]) / 2)
        
        axes[1, t].imshow(original_even[t], cmap='gray')
        axes[2, t].imshow(generated_even[t], cmap='gray')
        axes[1, t].axis('off')
        axes[2, t].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved reconstructions to {save_path}")
    
    save_frames(original_odd, generated_even)

def visualize_dataset_sample(dataset, method, sample_idx=0, save_path=None):
    """
    Visualize a single sample from the dataset, showing both the Fourier components
    and reconstructed images.
    """
    sequence = dataset[sample_idx]
    odd_frames = sequence[::2]  # Shape: (10, H, W)
    even_frames = sequence[1::2]  # Shape: (9, H, W)

    if method == 'interpolation' or method == 'gan':
        # Create figure for reconstructed images
        fig2, axes2 = plt.subplots(2, max(odd_frames.shape[0], even_frames.shape[0]), 
                                figsize=(20, 6))
        fig2.suptitle('Images', fontsize=16)
        
        # Plot reconstructed odd frames
        axes2[0, 0].set_ylabel('Odd Frames')
        for i in range(odd_frames.shape[0]):
            axes2[0, i].imshow(odd_frames[i], cmap='gray')
            axes2[0, i].axis('off')
        
        # Plot reconstructed even frames
        axes2[1, 0].set_ylabel('Even Frames')
        for i in range(even_frames.shape[0]):
            axes2[1, i].imshow(even_frames[i], cmap='gray')
            axes2[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig2.savefig(f'{save_path}_reconstructed.png', bbox_inches='tight', dpi=300)
    else: 
        # Create figure for Fourier components
        fig1, axes1 = plt.subplots(4, max(odd_frames.shape[0], even_frames.shape[0]), 
                                figsize=(20, 12))
        fig1.suptitle('Fourier Components', fontsize=16)
        
        # Plot odd frames
        axes1[0, 0].set_ylabel('Odd Magnitude')
        axes1[1, 0].set_ylabel('Odd Phase')
        for i in range(odd_frames.shape[0]):
            axes1[0, i].imshow(odd_frames[i, 0], cmap='viridis')
            axes1[1, i].imshow(odd_frames[i, 1], cmap='twilight')
            axes1[0, i].axis('off')
            axes1[1, i].axis('off')
        
        # Plot even frames
        axes1[2, 0].set_ylabel('Even Magnitude')
        axes1[3, 0].set_ylabel('Even Phase')
        for i in range(even_frames.shape[0]):
            axes1[2, i].imshow(even_frames[i, 0], cmap='viridis')
            axes1[3, i].imshow(even_frames[i, 1], cmap='twilight')
            axes1[2, i].axis('off')
            axes1[3, i].axis('off')
        
        # Create figure for reconstructed images
        fig2, axes2 = plt.subplots(2, max(odd_frames.shape[0], even_frames.shape[0]), 
                                figsize=(20, 6))
        fig2.suptitle('Reconstructed Images', fontsize=16)
        
        # Plot reconstructed odd frames
        axes2[0, 0].set_ylabel('Odd Frames')
        for i in range(odd_frames.shape[0]):
            recon_odd = reconstruct_image(odd_frames[i, 0], odd_frames[i, 1])
            axes2[0, i].imshow(recon_odd, cmap='gray')
            axes2[0, i].axis('off')
        
        # Plot reconstructed even frames
        axes2[1, 0].set_ylabel('Even Frames')
        for i in range(even_frames.shape[0]):
            recon_even = reconstruct_image(even_frames[i, 0], even_frames[i, 1])
            axes2[1, i].imshow(recon_even, cmap='gray')
            axes2[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig1.savefig(f'{save_path}_fourier.png', bbox_inches='tight', dpi=300)
            fig2.savefig(f'{save_path}_reconstructed.png', bbox_inches='tight', dpi=300)
    
    # plt.show()

def match_contrast(generated: torch.Tensor, reference: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    if generated.dim() == 3:
        generated = generated.squeeze(0)
    if reference.dim() == 3:
        reference = reference.squeeze(0)

    # Clone to avoid shared memory issues
    generated = generated.clone()
    reference = reference.clone()

    gen_flat = generated.flatten()
    ref_flat = reference.flatten()

    # Histograms
    gen_hist = torch.histc(gen_flat, bins=num_bins, min=0.0, max=1.0)
    ref_hist = torch.histc(ref_flat, bins=num_bins, min=0.0, max=1.0)

    # CDFs
    gen_cdf = torch.cumsum(gen_hist, dim=0)
    ref_cdf = torch.cumsum(ref_hist, dim=0)
    gen_cdf = gen_cdf / gen_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]

    # Mapping
    mapping = torch.zeros(num_bins)
    ref_bin_values = torch.linspace(0, 1, steps=num_bins)

    for i in range(num_bins):
        idx = torch.argmin(torch.abs(ref_cdf - gen_cdf[i]))
        mapping[i] = ref_bin_values[idx]

    # Digitize and apply mapping
    gen_indices = torch.clamp((gen_flat * (num_bins - 1)).long(), 0, num_bins - 1)
    matched_flat = mapping[gen_indices]

    matched = matched_flat.view_as(generated)

    return matched

def visualize_model_predictions(model, evaluator, dataset, device, method, sample_idx=0, save_dir='predictions'):
    """
    Generate and visualize model predictions for a single sample.
    """
    os.makedirs(save_dir, exist_ok=True)
    if method == 'gan':
        model.generator.eval()
    else:
        model.eval()
    
    # Get sample data
    # odd_frames, original_even_frames = dataset[sample_idx]
    # odd_frames = odd_frames.unsqueeze(0).to(device)
    # original_even_frames = original_even_frames.unsqueeze(0).to(device)
    sequence = dataset[sample_idx]
    sequence = sequence.unsqueeze(0).to(device)
    odd_frames = sequence[:, ::2]
    original_even_frames = sequence[:, 1::2]
    generated_frames = []
    
    with torch.no_grad():
        # Generate each even frame
        for t in range(original_even_frames.shape[1]):
            # Get surrounding odd frames as condition
            if method == 'unet':
                if t < original_even_frames.shape[1] - 1:
                    condition = torch.cat([odd_frames[:, t], odd_frames[:, t+1]], dim=1)
                else:
                    condition = torch.cat([odd_frames[:, t], odd_frames[:, t+1]], dim=1)
                 
                # Create time tensor
                time = torch.tensor([t / original_even_frames.shape[1]]).to(device)
                    
                # noise = (odd_frames[:, t] + odd_frames[:, t+1]) / 2
                # noise = torch.rand(original_even_frames[:, t].shape).to(device)
                # noise = original_even_frames[:, t]

                # Generate even frame
                # generated = model(noise, condition, time)
                generated = model(condition, time)
            elif method == 'interpolation':
                frame1 = odd_frames[:, t].unsqueeze(1)
                frame2 = odd_frames[:, t+1].unsqueeze(1)
                generated = model(frame1, frame2)
            elif method == 'gan':
                frame1 = odd_frames[:, t].unsqueeze(1)
                frame2 = odd_frames[:, t+1].unsqueeze(1)
                generated = model.generator(frame1, frame2)
            elif method == 'diffusion':
                condition = torch.cat([odd_frames[:, t], odd_frames[:, t+1]], dim=1)
                generated = sample_diffusion(model, condition, device, odd_frames[:, t].shape)

            generated_frames.append(generated.cpu().squeeze())

        for i in range(len(generated_frames)):
            generated_frames[i] = match_contrast(generated_frames[i], original_even_frames[0, i])

        generated_frames = torch.stack(generated_frames)
    
    original_even_frames = original_even_frames.squeeze().cpu()

    evaluation_results = evaluator.benchmark(original_even_frames, generated_frames)
    
    if method == 'interpolation' or method == 'gan':
        visualize_interpolations(
            odd_frames.squeeze().cpu().numpy(),
            original_even_frames.numpy(),
            generated_frames.numpy(),
            save_path=os.path.join(save_dir, f'sample_{sample_idx}_reconstructed.png'),
            eval_metrics=evaluation_results
        )
    else:
        # Visualize results
        visualize_results(
            odd_frames.squeeze().cpu().numpy(),
            original_even_frames.squeeze().cpu().numpy(),
            generated_frames.numpy(),
            save_path=os.path.join(save_dir, f'sample_{sample_idx}_fourier.png')
        )
        
        visualize_reconstructions(
            odd_frames.squeeze().cpu().numpy(),
            original_even_frames.squeeze().cpu().numpy(),
            generated_frames.numpy(),
            save_path=os.path.join(save_dir, f'sample_{sample_idx}_reconstructed.png')
        )

        # save_frames(
        #     odd_frames.squeeze().cpu().numpy(), 
        #     generated_frames.numpy(), 
        # )

def plot_losses(train_losses, test_losses, save_path=None):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, test_losses, 'g-', label='Test Loss', linewidth=2)
    
    plt.title('Training and Test Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_losses_gan(gen_losses, disc_losses, fft_disc_losses, save_path=None):
    """
    Visualizes Generator Loss, Discriminator Loss, and FFT Discriminator Loss on one plot.
    
    Parameters:
    - gen_losses (list): List of Generator losses.
    - disc_losses (list): List of Discriminator losses.
    - fft_disc_losses (list): List of FFT Discriminator losses.
    """
    epochs = range(1, len(gen_losses) + 1)  # Assuming the losses are recorded per epoch
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, gen_losses, label='Generator Loss', color='blue')
    plt.plot(epochs, disc_losses, label='Discriminator Loss', color='red')
    plt.plot(epochs, fft_disc_losses, label='FFT Discriminator Loss', color='green')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Generator, Discriminator and FFT Discriminator Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# A function to save the original odd frames and generated frames named in a sequence
def save_frames(original_odd, generated_even, save_dir='sequence_predictions'):
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(len(original_odd)):
        plt.imsave(f'{save_dir}/{(2*i)+1}.png', original_odd[i], cmap='gray')
    for i in range(len(generated_even)):
        plt.imsave(f'{save_dir}/{(i+1)*2}.png', generated_even[i], cmap='gray')
    
    print(f"Saved frames to {save_dir}")