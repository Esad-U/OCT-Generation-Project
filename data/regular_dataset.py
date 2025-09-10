import numpy as np
import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset

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

        # if len(image_files) != 19:
        #     raise ValueError(f"Folder {folder_path} contains {len(image_files)} images instead of 19.")

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
