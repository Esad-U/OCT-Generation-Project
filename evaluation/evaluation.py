"""
    Evaluation module for the OCT Generation model.
"""
import lpips
import cv2
import numpy as np

from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance

class Evaluator:
    def __init__(self, device='cpu'):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.lpips = lpips.LPIPS(net='alex').to(device)

        self.dataset_averages = {'ssim': 0, 'psnr': 0, 'brisque': 0}
        self.data_count = 0

    def compute_ssim(self, img1, img2):
        return self.ssim(img1, img2)

    def compute_psnr(self, img1, img2):
        return self.psnr(img1, img2)

    def compute_fid(self, real_images, generated_images):
        self.fid.update(real_images, real=True)
        self.fid.update(generated_images, real=False)
        return self.fid.compute()

    def compute_lpips(self, img1, img2):
        return self.lpips(img1, img2)

    def calculate_brisque_opencv(self, image_input):
        """
        Calculate BRISQUE score using OpenCV's built-in implementation
        Requires opencv-contrib-python
        
        Args:
            image_input: Can be:
                - str: path to image file
                - numpy array: image array (H, W, C) or (H, W)
                - torch.Tensor: PyTorch tensor (C, H, W) or (H, W, C) or (H, W)
        
        Returns:
            float: BRISQUE score (lower is better)
        """
        # Handle different input types
        if isinstance(image_input, str):
            # File path
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not load image from {image_input}")
                
        elif hasattr(image_input, 'numpy'):
            # PyTorch tensor
            import torch
            
            # Convert tensor to numpy
            if image_input.requires_grad:
                img_np = image_input.detach().cpu().numpy()
            else:
                img_np = image_input.cpu().numpy()
            
            # Handle different tensor formats
            if len(img_np.shape) == 4:
                # Batch dimension (B, C, H, W) - take first image
                img_np = img_np[0]
            
            if len(img_np.shape) == 3:
                if img_np.shape[0] <= 4:  # Likely (C, H, W) format
                    img_np = np.transpose(img_np, (1, 2, 0))  # Convert to (H, W, C)
                # else: already (H, W, C)
            
            # Convert to uint8 if needed
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:
                    # Normalized to [0, 1]
                    img_np = (img_np * 255).astype(np.uint8)
                elif img_np.max() <= 255:
                    # Already in [0, 255] range
                    img_np = img_np.astype(np.uint8)
                else:
                    # Need to normalize
                    img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
            
            # Handle grayscale
            if len(img_np.shape) == 2:
                img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 1:
                img = cv2.cvtColor(img_np.squeeze(2), cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 3:
                # Assume RGB, convert to BGR for OpenCV
                img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif img_np.shape[2] == 4:
                # RGBA, convert to BGR
                img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img = img_np
                
        else:
            # Numpy array or similar
            img = image_input
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                # Assume RGB, convert to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if img is None:
            raise ValueError("Could not process input image")
        
        # OpenCV BRISQUE implementation
        # Try different API versions
        try:
            # Method 2: Direct computation without model files
            score = cv2.quality.QualityBRISQUE_compute(img, "brisque_model_live.yml", "brisque_range_live.yml")[0]
            if score == 0:
                cv2.imwrite('image.png', img)
        except:
            # Method 3: Fall back to custom implementation
            print("OpenCV BRISQUE failed, using custom implementation")
        return score

    def benchmark(self, original_frames, generated_frames):
        average_metrics = {'ssim': 0, 'psnr': 0, 'fid': 0, 'lpips': 0, 'brisque': 0}
        for i in range(len(generated_frames)):
            img1 = original_frames[i].clone().unsqueeze(0).unsqueeze(0)
            img2 = generated_frames[i].clone().unsqueeze(0).unsqueeze(0)

            average_metrics['ssim'] += self.compute_ssim(img1, img2)
            average_metrics['psnr'] += self.compute_psnr(img1, img2)
            # average_metrics['fid'] += self.compute_fid(img1, img2)
            # average_metrics['lpips'] += self.compute_lpips(img1, img2)
            average_metrics['brisque'] += self.calculate_brisque_opencv(generated_frames[i])

        for k in average_metrics.keys():
            average_metrics[k] /= len(generated_frames)
        
        return average_metrics
    
    def benchmark_without_visualization(self, original_frames, generated_frames):
        for i in range(len(generated_frames)):
            img1 = original_frames[i].clone().unsqueeze(0).unsqueeze(0)
            img2 = generated_frames[i].clone().unsqueeze(0).unsqueeze(0)

            ssim = self.compute_ssim(img1, img2)
            psnr = self.compute_psnr(img1, img2)
            brisque_score = self.calculate_brisque_opencv(generated_frames[i])

            self.dataset_averages['ssim'] += ssim
            self.dataset_averages['psnr'] += psnr
            self.dataset_averages['brisque'] += brisque_score
            self.data_count += 1

            if self.data_count % 10 == 0:
                print(f"Sample {self.data_count} => SSIM: {ssim:.4f} - PSNR: {psnr:.2f} - BRISQUE: {brisque_score:.2f}")
        
        return True
