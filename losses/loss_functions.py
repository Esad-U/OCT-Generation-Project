import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils.utils import reconstruct_image
from kornia.losses import psnr_loss
from kornia.filters import sobel
from torchmetrics import StructuralSimilarityIndexMeasure

# TODO: Daha detaylı bakacak bir loss fonksiyonuna ihtiyacım var
# TODO: SSIM - Deneniyor Kornia ile de denenebilir -> kornia.losses.ssim_loss
# TODO: Gradient loss: Sobel + MSE - yazdım deniyorum
# TODO: PSNR - yazdım denemedim
# TODO: Perceptual loss with VGG
# TODO: SSIM + L1 - yazdım denemedim

class PerceptualStyleLoss(nn.Module):
    def __init__(self,
                 content_layers=['conv3_2'],
                 style_layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3'],
                 content_weights=None,
                 style_weights=None,
                 device='cuda'):
        super().__init__()

        # Load pretrained VGG19 and extract features
        vgg = models.vgg19(pretrained=True).features.to(device).eval()

        self.vgg_layers = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_4': 16,
            'conv4_1': 19, 'conv4_2': 21, 'conv4_3': 23, 'conv4_4': 25,
            'conv5_1': 28, 'conv5_2': 30, 'conv5_3': 32, 'conv5_4': 34
        }

        self.content_layers = content_layers
        self.style_layers = style_layers

        # Layer importance weights
        self.content_weights = content_weights or {name: 1.0 for name in content_layers}
        self.style_weights = style_weights or {name: 1.0 for name in style_layers}
        
        # Create separate modules for each layer we need to access
        self.feature_modules = nn.ModuleDict()
        for name in set(content_layers + style_layers):
            idx = self.vgg_layers[name]
            # Create a sequential module up to the required layer
            self.feature_modules[name] = nn.Sequential(*list(vgg.children())[:idx+1])
            
        # Freeze VGG parameters
        for module in self.feature_modules.values():
            for param in module.parameters():
                param.requires_grad = False

        self.criterion = nn.L1Loss()
        self.device = device

    def gram_matrix(self, feat):
        B, C, H, W = feat.shape
        feat = feat.view(B, C, H * W)
        G = torch.bmm(feat, feat.transpose(1, 2))  # (B, C, C)
        return G / (C * H * W)

    def forward(self, generated, target):
        # Handle grayscale inputs
        if generated.shape[1] == 1:
            generated = generated.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)

        perceptual_loss = 0.0
        style_loss = 0.0

        # Process content loss
        for layer_name in self.content_layers:
            weight = self.content_weights[layer_name]
            module = self.feature_modules[layer_name]
            
            gen_features = module(generated)
            target_features = module(target)
            
            layer_loss = self.criterion(gen_features, target_features)
            perceptual_loss += weight * layer_loss

        # Process style loss
        for layer_name in self.style_layers:
            weight = self.style_weights[layer_name]
            module = self.feature_modules[layer_name]
            
            gen_features = module(generated)
            target_features = module(target)
            
            gen_gram = self.gram_matrix(gen_features)
            target_gram = self.gram_matrix(target_features)
            
            layer_loss = self.criterion(gen_gram, target_gram)
            style_loss += weight * layer_loss

        return perceptual_loss, style_loss

class PerceptualLoss(nn.Module):
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_2'], device='cuda'):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16 model
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        
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

        self.criterion = nn.L1Loss()

    def forward(self, generated, target):
        """
        Computes perceptual loss between generated and real images.
        :param generated: Generated frame (B, C, H, W)
        :param target: Real frame (B, C, H, W)
        """
        loss = 0.0

        x = generated.repeat(1, 3, 1, 1)
        y = target.repeat(1, 3, 1, 1)

        count = 0

        for name, layer in enumerate(self.feature_extractor):
            x = layer(x)
            y = layer(y)

            if name in self.vgg_layers.values():
                loss += self.criterion(x, y)
                count += 1
        
        return loss / count

class PerceptualLossNovel(nn.Module):
    """
    Lightweight perceptual loss using shallow VGG19 layers for fine detail preservation.

    Uses:
        - relu1_1  (features[1])
        - relu2_1  (features[6])
    """
    def __init__(self, resize=False):
        super(PerceptualLossNovel, self).__init__()

        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # Extract only necessary layers
        self.relu1_1 = nn.Sequential(*[vgg[i] for i in range(0, 2)])   # Conv1 + ReLU
        self.relu2_1 = nn.Sequential(*[vgg[i] for i in range(2, 7)])   # Conv + ReLU

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        self.resize = resize

    def forward(self, pred, target):
        # VGG expects 3-channel input; repeat grayscale if needed
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Optional resizing to VGG input size (224x224)
        if self.resize:
            pred = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        # Forward through selected layers
        pred_relu1_1 = self.relu1_1(pred)
        target_relu1_1 = self.relu1_1(target)
        pred_relu2_1 = self.relu2_1(pred_relu1_1)
        target_relu2_1 = self.relu2_1(target_relu1_1)

        # Compute perceptual losses
        loss1 = F.l1_loss(pred_relu1_1, target_relu1_1)
        loss2 = F.l1_loss(pred_relu2_1, target_relu2_1)

        # Average
        return (loss1 + loss2) * 0.5

def novel_loss(outputs, target, perceptual_model=PerceptualLossNovel(resize=True).to('cuda')):
    weights = {
        'l1': 1.0,
        'ssim': 0.3,
        'perceptual': 0.1,
        'gradient': 0.1,
        'fft': 0.05,
        'alpha2': 0.3,
        'alpha3': 0.1,
    }

    # Main loss terms
    loss_l1 = nn.L1Loss()(outputs['main'], target)
    loss_ssim = ssim_loss(outputs['main'], target)
    loss_perceptual = perceptual_model(outputs['main'], target)
    loss_grad = gradient_loss(outputs['main'], target)
    loss_fft = fft_loss(outputs['main'], target)

    l_main = (
        weights['l1'] * loss_l1 +
        weights['ssim'] * loss_ssim +
        weights['perceptual'] * loss_perceptual +
        weights['gradient'] * loss_grad +
        weights['fft'] * loss_fft
    )

    # Deep supervision (only L1 for stability)
    l_ds2 = weights['alpha2'] * nn.L1Loss()(outputs['ds2'], target)
    l_ds3 = weights['alpha3'] * nn.L1Loss()(outputs['ds3'], target)

    total = l_main + l_ds2 + l_ds3
    return total, {"main": l_main.item(), "ds2": l_ds2.item(), "ds3": l_ds3.item()}

def film_loss(pred, target, perceptual_gram, weights = [1, 1, 1]):
    l1 = nn.L1Loss()(pred, target)
    perceptual, gram = perceptual_gram(pred, target)

    return weights[0] * l1 + weights[1] * perceptual + weights[2] * gram

def fft_loss(pred, target):
    """
    Compute frequency-domain loss between predicted and target images.
    Focuses on high-frequency details where fine structures live.
    """
    # Apply FFT
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)

    # Shift zero-frequency component to center
    pred_fft_shift = torch.fft.fftshift(pred_fft)
    target_fft_shift = torch.fft.fftshift(target_fft)

    # Compute magnitude spectrum (ignore phase)
    pred_mag = torch.abs(pred_fft_shift)
    target_mag = torch.abs(target_fft_shift)

    # Normalize to avoid scale dominance
    pred_mag = pred_mag / (torch.max(pred_mag) + 1e-8)
    target_mag = target_mag / (torch.max(target_mag) + 1e-8)

    # Use L1 loss in frequency domain
    return F.l1_loss(pred_mag, target_mag)

def gradient_loss(pred, target):
    # pred and target are of shape (B, H, W)
    # transform them into (B, C, H, W)

    pred_gradients = sobel(pred)
    target_gradients = sobel(target)

    gradient_l = nn.MSELoss()(pred_gradients, target_gradients)

    return gradient_l

def create_window(window_size, channel=1):
    def gaussian(window_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - window_size//2)**2/float(2*sigma**2) 
                          for x in range(window_size)]))
        return gauss/gauss.sum()
    
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_loss(pred, target):
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
    score = ssim_metric(pred, target)
    loss = 1 - score
    
    return loss

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
