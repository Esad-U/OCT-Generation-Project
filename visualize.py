import torch
import matplotlib.pyplot as plt
import os

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
    
    plt.show()

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
    
    plt.show()

def visualize_interpolations(original_odd, original_even, generated_even, save_path=None):
    """Visualize reconstructed images"""
    num_timesteps = original_even.shape[0]
    fig, axes = plt.subplots(3, num_timesteps, figsize=(20, 8))
    
    # Plot titles
    axes[0, num_timesteps//2].set_title("Odd Frames", pad=10)
    axes[1, num_timesteps//2].set_title("Original Even Frames", pad=10)
    axes[2, num_timesteps//2].set_title("Generated Even Frames", pad=10)
    
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
    odd_frames, even_frames = dataset[sample_idx]

    if method == 'interpolation':
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
    
    plt.show()

def visualize_model_predictions(model, dataset, device, method, sample_idx=0, save_dir='predictions'):
    """
    Generate and visualize model predictions for a single sample.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Get sample data
    odd_frames, original_even_frames = dataset[sample_idx]
    odd_frames = odd_frames.unsqueeze(0).to(device)
    original_even_frames = original_even_frames.unsqueeze(0).to(device)
    
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
                
                noise = (odd_frames[:, t] + odd_frames[:, t+1]) / 2
                # noise = torch.rand(original_even_frames[:, t].shape).to(device)
                # noise = original_even_frames[:, t]

                # Generate even frame
                generated = model(noise, condition, time)
            elif method == 'interpolation':
                frame1 = odd_frames[:, t]
                frame2 = odd_frames[:, t+1]
                generated = model(frame1, frame2)
            elif method == 'diffusion':
                condition = torch.cat([odd_frames[:, t], odd_frames[:, t+1]], dim=1)
                generated = sample_diffusion(model, condition, device, odd_frames[:, t].shape)

            generated_frames.append(generated.cpu().squeeze())
    
    generated_frames = torch.stack(generated_frames)
    
    if method == 'interpolation':
        visualize_interpolations(
            odd_frames.squeeze().cpu().numpy(),
            original_even_frames.squeeze().cpu().numpy(),
            generated_frames.numpy(),
            save_path=os.path.join(save_dir, f'sample_{sample_idx}_reconstructed.png')
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

# A function to save the original odd frames and generated frames named in a sequence
def save_frames(original_odd, generated_even, save_dir='sequence_predictions'):
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(len(original_odd)):
        plt.imsave(f'{save_dir}/{(2*i)+1}.png', original_odd[i], cmap='gray')
    for i in range(len(generated_even)):
        plt.imsave(f'{save_dir}/{(i+1)*2}.png', generated_even[i], cmap='gray')
    
    print(f"Saved frames to {save_dir}")