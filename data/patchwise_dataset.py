import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F

class PatchwiseDataset(Dataset):
    def __init__(self, image_dir, patch_size=128, window_size=3, train_mode=True, background_threshold=0.05, transform=None):
        """
        Args:
            patch_size: Size of the crop for training (e.g., 128 or 256).
            train_mode: If True, returns patches. If False, returns full images.
            background_threshold: Minimum mean pixel intensity (0-1) to accept a patch.
        """
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.window_size = window_size
        self.transform = transform
        self.train_mode = train_mode
        self.background_threshold = background_threshold
        
        self.samples = []
        
        all_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_' in f
        ]

        # Grouping by prefix (e.g. "1" from "1_01.png")
        group_dict = {}
        for f in all_files:
            group_id = f.split('_')[0]
            group_dict.setdefault(group_id, []).append(f)

        # Collect sliding windows
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
        # 1. Load the Full Sequence (3 frames)
        # Shape: (3, H, W)
        full_sequence = self._load_sequence_raw(self.samples[idx])

        # 2. TRAINING MODE: Patch-wise with Background Rejection
        if self.train_mode:
            c, h, w = full_sequence.shape
            
            # Safety check: if image is smaller than patch, return center crop without loop
            if h <= self.patch_size or w <= self.patch_size:
                return self._center_crop(full_sequence)

            # Try up to 10 times to find a non-black patch
            for _ in range(10):
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                
                patch = full_sequence[:, top:top+self.patch_size, left:left+self.patch_size]
                
                # Check mean intensity of the middle frame (target)
                # If it's brighter than threshold, we accept it.
                if patch[1].mean() > self.background_threshold:
                    return patch
            
            # If we fail 10 times (very unlikely unless image is pure black), return the last patch
            return patch

        # 3. INFERENCE MODE: Full Image with Padding
        else:
            # We must pad to be divisible by 32 (standard for U-Net with 5 pools)
            # otherwise concatenation errors occur in the decoder.
            return self._pad_to_divisible(full_sequence, divisor=32)

    def _load_sequence_raw(self, image_paths):
        # Loads images at ORIGINAL resolution (no resizing here)
        sequence = []
        for p in image_paths:
            # Load and normalize to 0-1
            img = Image.open(p).convert('L')
            img = np.array(img) / 255.0
            sequence.append(img)
            
        sequence = np.stack(sequence, axis=0)
        tensor = torch.tensor(sequence, dtype=torch.float32)
        
        # Apply transforms if they exist (usually normalization)
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor

    def _pad_to_divisible(self, tensor, divisor=32):
        """Pads the input tensor so height and width are divisible by 'divisor'."""
        _, h, w = tensor.shape
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        
        if pad_h == 0 and pad_w == 0:
            return tensor
            
        # F.pad expects (left, right, top, bottom)
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

    def _center_crop(self, tensor):
        _, h, w = tensor.shape
        th, tw = self.patch_size, self.patch_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return tensor[:, i:i+th, j:j+tw]