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


class RegularDataset(Dataset):
    def __init__(self, root_dir, image_size=128, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.folders = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, folder))]
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder_path = self.folders[idx]
        image_files = sorted([os.path.join(folder_path, file) 
                            for file in os.listdir(folder_path) 
                            if file.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if len(image_files) != 19:
            raise ValueError(f"Folder {folder_path} contains {len(image_files)} images instead of 19.")

        sequence = []
        
        for image_file in image_files:
            image = Image.open(image_file).convert('L')
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image) / 255.0

            sequence.append(image)

        sequence = np.stack(sequence, axis=0)  # Shape: (19, H, W)
        sequence = torch.tensor(sequence, dtype=torch.float32)
        
        # Split into odd and even frames
        # odd_frames = sequence[::2]  # Shape: (10, H, W)
        # even_frames = sequence[1::2]  # Shape: (9, H, W)

        return sequence


class EfficientDataset(Dataset):
    def __init__(self, image_dir, image_size=128, window_size=3, transform=None):
        self.image_dir = image_dir
        self.image_size = image_size
        self.window_size = window_size
        self.transform = transform

        # Gather all image files
        all_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_' in f
        ]

        # Group files by their folder prefix (e.g., "1" from "1_01.png")
        group_dict = {}
        for f in all_files:
            prefix = f.split('_')[0]
            if prefix not in group_dict:
                group_dict[prefix] = []
            group_dict[prefix].append(f)

        # Sort files within each group and prepare samples
        self.samples = []
        for group_id in sorted(group_dict.keys(), key=lambda x: int(x)):
            files = sorted(group_dict[group_id])
            if len(files) >= window_size:
                window = files[:window_size]
                full_paths = [os.path.join(image_dir, fname) for fname in window]
                self.samples.append(full_paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths = self.samples[idx]
        sequence = []

        for image_file in image_paths:
            image = Image.open(image_file).convert('L')
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image) / 255.0
            sequence.append(image)

        sequence = np.stack(sequence, axis=0)  # Shape: (3, H, W)
        sequence = torch.tensor(sequence, dtype=torch.float32)

        if self.transform:
            sequence = self.transform(sequence)

        return sequence