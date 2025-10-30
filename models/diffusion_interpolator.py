import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import random
import os

from functools import partial
from torchvision.utils import make_grid, save_image

# -------------------------
# Utilities
# -------------------------
def sinusoidal_time_embedding(timesteps, dim):
    """
    timesteps: (B,) tensor of integer timesteps
    returns: (B, dim) time embeddings
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)

def default(device=None, dtype=None, val=None):
    if val is None: return None
    return val.to(device=device, dtype=dtype)

# -------------------------
# Building blocks (reuse style from earlier)
# -------------------------
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H*W)
        k = self.key(x).view(B, -1, H*W)
        v = self.value(x).view(B, -1, H*W)
        attn = torch.bmm(q.permute(0,2,1), k)
        attn = F.softmax(attn / (q.size(1)**0.5), dim=-1)
        out = torch.bmm(v, attn.permute(0,2,1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.need_proj = (in_ch != out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        if self.need_proj:
            self.proj = nn.Conv2d(in_ch, out_ch, 1)
        # optional time embedding -> affine modulation
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, out_ch),
                nn.ReLU(),
                nn.Linear(out_ch, out_ch)
            )
        else:
            self.time_mlp = None

    def forward(self, x, t_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        if self.time_mlp is not None and t_emb is not None:
            # t_emb: (B, dim) -> (B, out_ch) -> (B, out_ch, 1,1)
            shift = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + shift
        h = self.conv2(h)
        h = self.norm2(h)
        res = self.proj(x) if self.need_proj else x
        return self.act(h + res)

# -------------------------
# UNet denoiser conditioned on two frames
# -------------------------
class ConditionalUNet(nn.Module):
    """
    Takes (noisy_target, cond_frames) concatenated as channels input,
    where cond_frames = [frame1, frame2] (each 1 channel).
    Architecture: encoder -> bridge (attention) -> decoder with skip connections.
    Predicts noise of same shape as the target (1 channel).
    """
    def __init__(self, base_ch=48, in_channels=3,  # 1 noisy target + 2 cond => 3
                 out_channels=1, time_emb_dim=256, use_attention=True):
        super().__init__()
        self.base_ch = base_ch
        self.time_emb_dim = time_emb_dim
        # initial conv
        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        # time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # encoder blocks
        self.enc1 = ResidualBlock(base_ch, base_ch, time_emb_dim)
        self.enc2 = ResidualBlock(base_ch, base_ch*2, time_emb_dim)
        self.enc3 = ResidualBlock(base_ch*2, base_ch*4, time_emb_dim)

        # pooling
        self.pool = nn.AvgPool2d(2)

        # bottleneck
        self.bottleneck = ResidualBlock(base_ch*4, base_ch*8, time_emb_dim)
        self.attn = SelfAttention(base_ch*8) if use_attention else nn.Identity()

        # decoder blocks
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ResidualBlock(base_ch*8 + base_ch*4, base_ch*4, time_emb_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ResidualBlock(base_ch*4 + base_ch*2, base_ch*2, time_emb_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ResidualBlock(base_ch*2 + base_ch, base_ch, time_emb_dim)

        # final conv to predict noise for the *target* channel (1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, out_channels, kernel_size=1)
        )

    def forward(self, noisy_target, cond_frames, t):
        """
        noisy_target: (B,1,H,W)
        cond_frames: (B,2,H,W) [frame1, frame2]
        t: (B,) integer timesteps
        """
        # time embedding
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)  # (B, time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        x = torch.cat([noisy_target, cond_frames], dim=1)  # (B,3,H,W)
        x = self.init_conv(x)  # (B, base_ch, H, W)

        e1 = self.enc1(x, t_emb)   # base_ch, H
        e2 = self.enc2(self.pool(e1), t_emb)  # base_ch*2, H/2
        e3 = self.enc3(self.pool(e2), t_emb)  # base_ch*4, H/4

        b = self.bottleneck(self.pool(e3), t_emb)  # base_ch*8, H/8
        b = self.attn(b)

        d3 = self.up3(b)  # H/4
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3, t_emb)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, t_emb)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, t_emb)

        eps_pred = self.final_conv(d1)  # predict noise for the target channel
        return eps_pred  # (B,1,H,W)


# -------------------------
# Gaussian diffusion utilities
# -------------------------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class GaussianDiffusion:
    def __init__(self, model: ConditionalUNet, timesteps=1000, device='cpu',
                 p_uncond=0.1):
        """
        model: conditional denoiser
        timesteps: number of diffusion steps
        p_uncond: probability to drop conditioning during training (classifier-free guidance)
        """
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.p_uncond = p_uncond

        betas = linear_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)

        self.register_buffer = lambda name, val: setattr(self, name, val)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # precompute useful terms
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    # helper to add buffers as attributes (we used register_buffer substitute above)
    # training-time forward diffusion (q_sample)
    def q_sample(self, x_start, t, noise=None):
        """
        x_start: (B,1,H,W) in [0,1]
        t: (B,) timesteps
        noise: optional noise, else standard normal
        returns x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om_acp = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_acp * x_start + sqrt_om_acp * noise

    # training step: compute loss (predict noise)
    def p_losses(self, x_start, cond_frames, t, noise=None, l1_weight=0.0):
        """
        x_start: (B,1,H,W) ground truth middle frame in [0,1]
        cond_frames: (B,2,H,W) frames [f1,f2]
        t: (B,) timesteps
        noise: (B,1,H,W) optional
        returns: mse loss (and optionally l1 between reconstructed x0 and x_start)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        # classifier-free guidance: with p_uncond probability drop conditioning during training
        do_uncond = (torch.rand(x_start.shape[0], device=x_start.device) < self.p_uncond).long()
        # prepare cond input: if dropped -> zeros
        cond_for_model = cond_frames.clone()
        cond_for_model[do_uncond == 1] = 0.0

        print(x_t.shape)
        print(cond_for_model.shape)

        # model predicts noise
        eps_pred = self.model(x_t, cond_for_model, t)

        # simple L2 loss
        loss_mse = F.mse_loss(eps_pred, noise)

        # optional reconstruction L1 loss on x0
        if l1_weight > 0:
            # predict x0 from predicted eps: x0_pred = (x_t - sqrt(1-alpha_cumprod)*eps_pred)/sqrt(alpha_cumprod)
            sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_om_acp = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            x0_pred = (x_t - sqrt_om_acp * eps_pred) / sqrt_acp
            loss_l1 = F.l1_loss(x0_pred.clamp(0.0, 1.0), x_start)
        else:
            loss_l1 = torch.tensor(0.0, device=x_start.device)

        return loss_mse + l1_weight * loss_l1, {'mse': loss_mse.item(), 'l1': loss_l1.item() if isinstance(loss_l1, torch.Tensor) else float(loss_l1)}

    # DDPM sampling (with optional classifier-free guidance)
    @torch.no_grad()
    def sample(self, cond_frames, shape=(1,1,128,128), guidance_weight=0.0, device=None, progress=False):
        """
        cond_frames: (B,2,H,W)
        shape: output shape, should match cond_frames spatial size
        guidance_weight: 0 -> no guidance (unconditional), >0 -> apply classifier-free guidance
        Returns generated x0 in [0,1]
        """
        device = device or self.device
        B = cond_frames.shape[0]
        x_t = torch.randn(shape, device=device)  # start from pure noise
        for t_ in reversed(range(self.timesteps)):
            t = torch.full((B,), t_, dtype=torch.long, device=device)

            if guidance_weight == 0.0:
                # normal conditional prediction
                eps_pred = self.model(x_t, cond_frames, t)
            else:
                # classifier-free guidance: get conditional and unconditional predictions
                # unconditional cond = zeros
                cond_zeros = torch.zeros_like(cond_frames)
                eps_cond = self.model(x_t, cond_frames, t)       # (B,1,H,W)
                eps_uncond = self.model(x_t, cond_zeros, t)      # (B,1,H,W)
                eps_pred = eps_uncond + guidance_weight * (eps_cond - eps_uncond)

            # compute posterior mean and variance as DDPM
            beta_t = self.betas[t_].view(1,1,1,1)
            sqrt_one_minus_acp = self.sqrt_one_minus_alphas_cumprod[t_].view(1,1,1,1)
            sqrt_recip_acp = self.sqrt_recip_alphas_cumprod[t_].view(1,1,1,1)

            # predict x0 from predicted eps
            x0_pred = (x_t - sqrt_one_minus_acp * eps_pred) / (self.sqrt_alphas_cumprod[t_].view(1,1,1,1))
            x0_pred = x0_pred.clamp(0.0, 1.0)

            # compute mean of q(x_{t-1} | x_t, x0_pred)
            coef1 = (self.betas[t_] * torch.sqrt(self.alphas_cumprod_prev[t_])) / (1.0 - self.alphas_cumprod[t_])
            coef2 = ((1.0 - self.alphas_cumprod_prev[t_]) * torch.sqrt(self.alphas[t_])) / (1.0 - self.alphas_cumprod[t_])
            posterior_mean = coef1.view(1,1,1,1) * x0_pred + coef2.view(1,1,1,1) * x_t

            if t_ > 0:
                var = self.posterior_variance[t_].view(1,1,1,1)
                noise = torch.randn_like(x_t)
                x_t = posterior_mean + torch.sqrt(var) * noise
            else:
                x_t = posterior_mean

        return x_t.clamp(0.0, 1.0)  # x0 final

# -------------------------
# Example training step for one batch
# -------------------------
def diffusion_train_step(batch, diffusion: GaussianDiffusion, optim, device,
                         l1_weight=0.0):
    """
    batch: dict with 'frame1', 'frame2', 'target' (all in [0,1], tensors)
    diffusion: GaussianDiffusion object
    optim: optimizer for diffusion.model.parameters()
    """
    frame1 = batch["frame1"].to(device)
    frame2 = batch["frame2"].to(device)
    target = batch["target"].to(device)

    B = target.shape[0]
    t = torch.randint(0, diffusion.timesteps, (B,), device=device).long()
    loss, metrics = diffusion.p_losses(target, torch.cat([frame1, frame2], dim=1), t, l1_weight=l1_weight)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item(), metrics

# -------------------------
# Usage & instantiation example
# -------------------------
# ----------------------
# Training loop
# ----------------------
# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("train_diffusion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_diffusion(
    dataloader,
    ckptdir="checkpoints/diffusion_interp",
    epochs=100,
    lr=2e-4,
    img_size=128,
    timesteps=1000,
    device="cuda",
    sample_interval=5,
    guidance=2.0,
    l1_weight=0.0,
):

    # model & diffusion
    model = ConditionalUNet(base_ch=48, in_channels=3, out_channels=1).to(device)
    diffusion = GaussianDiffusion(model, timesteps=timesteps, device=device, p_uncond=0.1)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))

    global_step = 0
    train_losses, mse_losses, l1_losses = [], [], []

    for epoch in range(epochs):
        model.train()
        logger.info(f"Epoch {epoch+1}/{epochs}")

        for batch_idx, sequence in enumerate(dataloader):
            f1, tgt, f2 = sequence[:, 0], sequence[:, 1], sequence[:, 2]  # all (B,1,H,W)
            batch_dict = {"frame1": f1, "frame2": f2, "target": tgt}

            loss, metrics = diffusion_train_step(batch_dict, diffusion, optim, device, l1_weight=l1_weight)

            # track metrics
            train_losses.append(loss)
            mse_losses.append(metrics["mse"])
            l1_losses.append(metrics["l1"])

            if batch_idx % 50 == 0:  # log every N steps
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Step {batch_idx}/{len(dataloader)} "
                    f"| Loss={loss:.4f} | MSE={metrics['mse']:.4f} | L1={metrics['l1']:.4f}"
                )

            global_step += 1

        # save checkpoint
        ckpt_path = os.path.join(ckptdir, f"model_epoch{epoch+1}.pt")
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }, ckpt_path)
        logger.info(f"Checkpoint saved at {ckpt_path}")

        # periodic sampling
        if (epoch+1) % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                seq = next(iter(dataloader))
                f1, tgt, f2 = seq[:, 0].to(device), seq[:, 1].to(device), seq[:, 2].to(device)
                cond = torch.cat([f1, f2], dim=1)
                samples = diffusion.sample(
                    cond, shape=(f1.size(0),1,img_size,img_size),
                    guidance_weight=guidance, device=device
                )

                # make grid: row1=f1, row2=tgt, row3=f2, row4=generated
                grid = torch.cat([f1, tgt, f2, samples], dim=0)
                grid = make_grid(grid, nrow=f1.size(0), normalize=True)
                save_path = os.path.join(ckptdir, f"samples_epoch{epoch+1}.png")
                save_image(grid, save_path)
                logger.info(f"Saved sample grid at {save_path}")

    # plot final training curves
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label="Total Loss", alpha=0.7)
    plt.plot(mse_losses, label="MSE Loss", alpha=0.7)
    plt.plot(l1_losses, label="L1 Loss", alpha=0.7)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Diffusion Training Losses")
    plt.savefig(os.path.join(ckptdir, "loss_curve.png"))
    logger.info("Saved training loss plot at loss_curve.png")