"""
    Evaluation module for the OCT Generation model.
"""
import lpips

from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance

class Evaluator:
    def __init__(self, device='cpu'):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.lpips = lpips.LPIPS(net='alex').to(device)

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

    def benchmark(self, original_frames, generated_frames):
        average_metrics = {'ssim': 0, 'psnr': 0, 'fid': 0, 'lpips': 0}
        for i in range(len(generated_frames)):
            img1 = original_frames[i].clone().unsqueeze(0).unsqueeze(0)
            img2 = generated_frames[i].clone().unsqueeze(0).unsqueeze(0)
            average_metrics['ssim'] += self.compute_ssim(img1, img2)
            average_metrics['psnr'] += self.compute_psnr(img1, img2)
            # average_metrics['fid'] += self.compute_fid(img1, img2)
            average_metrics['lpips'] += self.compute_lpips(img1, img2)

        for k in average_metrics.keys():
            average_metrics[k] /= len(generated_frames)
        
        return average_metrics
