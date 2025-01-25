import torch
import torch.nn as nn

def combined_loss(pred, target, fourier_weight=0.5):
    # Spatial domain loss
    pred_tmp = pred.cpu().detach().numpy()
    target_tmp = target.cpu().detach().numpy()
    reconstructed_pred = torch.from_numpy(reconstruct_image(pred_tmp[:, 0], pred_tmp[:, 1]))
    reconstructed_target = torch.from_numpy(reconstruct_image(target_tmp[:, 0], target_tmp[:, 1]))
    spatial_loss = nn.MSELoss()(reconstructed_pred, reconstructed_target)
    
    # Fourier domain loss
    fourier_loss = nn.MSELoss()(pred, target)
    
    # Combine both losses
    total_loss = (1 - fourier_weight) * spatial_loss + fourier_weight * fourier_loss
    return total_loss

def separate_loss(pred, target, phase_weight=0.5):
    # Magnitude loss
    mag_loss = nn.MSELoss()(pred[:, 0], target[:, 0])
    
    # Phase loss
    phase_loss = nn.MSELoss()(pred[:, 1], target[:, 1])
    
    # Combine both losses
    total_loss = (1 - phase_weight) * mag_loss + phase_weight * phase_loss
    return total_loss

