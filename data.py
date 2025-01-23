import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class ComplexFourierDataset(Dataset):
    def __init__(self, root_dir, image_size=128, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.folders = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, folder))]
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def fft_transform(self, image):
        """Convert image to Fourier domain, return magnitude and phase"""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Separate magnitude and phase
        magnitude = np.abs(f_shift)
        phase = np.angle(f_shift)
        
        # Log-scale magnitude
        magnitude = np.log1p(magnitude)  # Using log1p for numerical stability => ln(1 + x)
        
        # Normalize magnitude to [-1, 1]
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 2 - 1
        
        # Normalize phase to [-1, 1] (from [-π, π])
        phase = phase / np.pi
        
        return magnitude, phase

    def __getitem__(self, idx):
        folder_path = self.folders[idx]
        image_files = sorted([os.path.join(folder_path, file) 
                            for file in os.listdir(folder_path) 
                            if file.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if len(image_files) != 19:
            raise ValueError(f"Folder {folder_path} contains {len(image_files)} images instead of 19.")

        fourier_sequence = []
        
        for image_file in image_files:
            image = Image.open(image_file).convert('L')
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image) / 255.0
            
            magnitude, phase = self.fft_transform(image)
            # Stack magnitude and phase along new dimension
            fourier_data = np.stack([magnitude, phase], axis=0)
            fourier_sequence.append(fourier_data)

        fourier_sequence = np.stack(fourier_sequence, axis=0)  # Shape: (19, 2, H, W)
        fourier_sequence = torch.tensor(fourier_sequence, dtype=torch.float32)
        
        # Split into odd and even frames
        odd_frames = fourier_sequence[::2]  # Shape: (10, 2, H, W)
        even_frames = fourier_sequence[1::2]  # Shape: (9, 2, H, W)

        return odd_frames, even_frames