import numpy as np
import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset

class EfficientDataset(Dataset):
    def __init__(self, image_dir, image_size=128, window_size=3, transform=None):
        self.image_dir = image_dir
        self.image_size = image_size
        self.window_size = window_size
        self.transform = transform
        self.samples = []
        self.all_images = []

        all_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_' in f
        ]

        # Keep track of all full image paths for single-image random sampling
        self.all_images = [os.path.join(image_dir, f) for f in all_files]

        # Grouping by prefix (e.g. "1" from "1_01.png")
        group_dict = {}
        for f in all_files:
            group_id = f.split('_')[0]
            group_dict.setdefault(group_id, []).append(f)

        # Collect sliding windows for sequential samples
        for group_id in sorted(group_dict.keys(), key=lambda x: int(x)):
            files = sorted(group_dict[group_id])
            if len(files) < window_size:
                continue
            for i in range(len(files) - window_size + 1):
                window = files[i:i + window_size]
                full_paths = [os.path.join(image_dir, f) for f in window]
                self.samples.append(full_paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._load_sequence(self.samples[idx])

    def _load_sequence(self, image_paths):
        sequence = [self._load_image(p) for p in image_paths]
        sequence = np.stack(sequence, axis=0)
        return torch.tensor(sequence, dtype=torch.float32)

    def _load_image(self, image_path):
        image = Image.open(image_path).convert('L')
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image) / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float32)
        return self.transform(image_tensor) if self.transform else image_tensor

    def sample_random_images(self, batch_size):
        """Return a batch of random single images."""
        paths = random.choices(self.all_images, k=batch_size)
        images = [self._load_image(p) for p in paths]
        batch = torch.stack(images, dim=0).unsqueeze(1)  # Shape: (B, H, W)
        return batch
