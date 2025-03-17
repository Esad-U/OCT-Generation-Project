import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils import reconstruct_image
from kornia.losses import psnr_loss
from kornia.filters import sobel

# TODO: Daha detaylı bakacak bir loss fonksiyonuna ihtiyacım var
# TODO: SSIM - Deneniyor Kornia ile de denenebilir -> kornia.losses.ssim_loss
# TODO: Gradient loss: Sobel + MSE - yazdım deniyorum
# TODO: PSNR - yazdım denemedim
# TODO: Perceptual loss with VGG
# TODO: SSIM + L1 - yazdım denemedim


class PerceptualLoss(nn.Module):
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_2'], device='cuda'):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16 model
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        
        # Select layers to extract features
        self.selected_layers = layers
        self.vgg_layers = {
            'conv1_2': 4,
            'conv2_2': 9,
            'conv3_2': 16,
            'conv4_2': 23,
            'conv5_2': 30
        }

        self.feature_extractor = nn.Sequential(*list(vgg.children())[:max(self.vgg_layers.values()) + 1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze VGG weights

        self.criterion = nn.MSELoss()

    def forward(self, generated, target):
        """
        Computes perceptual loss between generated and real images.
        :param generated: Generated frame (B, C, H, W)
        :param target: Real frame (B, C, H, W)
        """
        loss = 0.0
        x = generated.unsqueeze(1)
        y = target.unsqueeze(1)

        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        
        for name, layer in enumerate(self.feature_extractor):
            x = layer(x)
            y = layer(y)

            if name in self.vgg_layers.values():
                loss += self.criterion(x, y)
        
        mse_loss = nn.MSELoss()(generated, target)

        return 0.5 * loss + 0.5 * mse_loss


def gradient_loss(pred, target):
    # pred and target are of shape (B, H, W)
    # transform them into (B, C, H, W)
    pred = pred.unsqueeze(1)
    target = target.unsqueeze(1)

    pred_gradients = sobel(pred)
    target_gradients = sobel(target)

    gradient_l = nn.MSELoss()(pred_gradients, target_gradients)
    mse_l = nn.MSELoss()(pred, target)

    return 0.4 * gradient_l + 0.6 * mse_l

def gradient_ssim_loss(pred, target):
    ssim_l = ssim(pred, target)
    
    pred = pred.unsqueeze(1)
    target = target.unsqueeze(1)

    pred_gradients = sobel(pred)
    target_gradients = sobel(target)

    gradient_l = nn.MSELoss()(pred_gradients, target_gradients)

    return 0.5 * gradient_l + 0.5 * ssim_l

def psnr(pred, target, alpha=0.1):
    mse_l = nn.MSELoss()(pred, target)
    pred = pred / 255.0
    target = target / 255.0
    psnr_l = 10 * torch.log10(1.0 / mse_l)
    return mse_l - alpha * psnr_l

def ssim_mse(pred, target, window_size=11):
    mse = nn.MSELoss()(pred, target)
    ssim_l = ssim(pred, target)

    return ssim_l + mse

def create_window(window_size, channel=1):
    def gaussian(window_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - window_size//2)**2/float(2*sigma**2) 
                          for x in range(window_size)]))
        return gauss/gauss.sum()
    
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Add channel dimension
    img1 = img1.unsqueeze(1)  # Shape becomes (batch_size, 1, height, width)
    img2 = img2.unsqueeze(1)
    
    # Value range checking
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = window_size // 2
    (batch_size, channel, height, width) = img1.size()
    
    # If window is not provided, create it
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    # Constants for stability
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # Calculate SSIM
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return 1 - ret

def combined_loss(pred, target, fourier_weight=0.5):
    # Spatial domain loss
    pred_tmp = pred.cpu().detach().numpy()
    target_tmp = target.cpu().detach().numpy()
    reconstructed_pred = torch.from_numpy(reconstruct_image(pred_tmp[:, 0], pred_tmp[:, 1])).float()
    reconstructed_target = torch.from_numpy(reconstruct_image(target_tmp[:, 0], target_tmp[:, 1])).float()
    spatial_loss = ssim(reconstructed_pred, reconstructed_target)
    
    # Fourier domain loss
    fourier_loss = separate_loss(pred, target)
    
    # Combine both losses
    total_loss = (1 - fourier_weight) * spatial_loss + fourier_weight * fourier_loss
    return total_loss

def separate_loss(pred, target, phase_weight=0.5):
    # Magnitude loss
    mag_loss = nn.L1Loss()(pred[:, 0], target[:, 0])
    
    # Phase loss
    phase_loss = nn.MSELoss()(pred[:, 1], target[:, 1])
    
    # Combine both losses
    total_loss = (1 - phase_weight) * mag_loss + phase_weight * phase_loss
    return total_loss

def interpolation_loss(pred, target):
    return nn.MSELoss()(pred, target)
