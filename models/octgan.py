import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from torch.nn.utils import spectral_norm
from losses.loss_functions import PerceptualStyleLoss, film_loss
from .generic import SelfAttentionUpdated, SEBlock, ResidualBlock


class Generator(nn.Module):
    def __init__(self, hidden_channels=64, in_channels=2, out_channels=1, dropout=0.3):
        super().__init__()
        hc = hidden_channels

        # Encoder
        self.inc = ResidualBlock(in_channels, hc)  # H
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(hc, hc*2))   # H/2
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(hc*2, hc*4)) # H/4
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(hc*4, hc*8)) # H/8

        # Bridge
        self.bridge = nn.Sequential(
            ResidualBlock(hc*8, hc*8),
            nn.Dropout(p=dropout),
            SelfAttention(hc*8)
        )

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = ResidualBlock(hc*8 + hc*4, hc*4)
        self.se3 = SEBlock(hc*4)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = ResidualBlock(hc*4 + hc*2, hc*2)
        self.se2 = SEBlock(hc*2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = ResidualBlock(hc*2 + hc, hc)

        self.outc = nn.Conv2d(hc, out_channels, 1)
        self.final_act = nn.Sigmoid()  # assume targets normalized to [0,1]

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=1)  # (B,2,H,W)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.bridge(x4)

        u3 = self.up3(b)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.conv_up3(u3)
        u3 = self.se3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)
        u2 = self.se2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.conv_up1(u1)

        out = self.outc(u1)
        out = self.final_act(out)
        return out  # (B,1,H,W)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3,  # we will pass concatenation [frame1, frame2, target_or_fake]
                 base_channels=64, n_layers=4, use_spectral_norm=True):
        super().__init__()
        layers = []
        ch = base_channels
        inp_ch = in_channels
        for i in range(n_layers):
            out_ch = ch if i == 0 else min(ch * (2**i), 512)
            conv = nn.Conv2d(inp_ch, out_ch, kernel_size=4, stride=2, padding=1)
            if use_spectral_norm:
                conv = spectral_norm(conv)
            layers += [conv, nn.LeakyReLU(0.2, inplace=True)]
            inp_ch = out_ch
        # add a few more conv layers with stride=1
        layers += [
            spectral_norm(nn.Conv2d(inp_ch, min(inp_ch*2, 512), kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(min(inp_ch*2,512), 1, kernel_size=4, stride=1, padding=1))
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, frame1, frame2, target):
        """
        frame1, frame2: (B,1,H,W)
        target: real or fake (B,1,H,W)
        Concatenate along channel: (B,3,H,W)
        """
        x = torch.cat([frame1, frame2, target], dim=1)
        return self.model(x)  # (B,1,H_patch,W_patch) patch logits


class OCTGAN():
    def __init__(self, g_hidden, lrG=2e-4, lrD=2e-4, device='cuda'):
        self.netG = Generator(hidden_channels=g_hidden, in_channels=2, out_channels=1).to(device)
        self.netD = PatchDiscriminator(in_channels=3, base_channels=64, n_layers=4).to(device)

        self.optimG = optim.Adam(self.netG.parameters(), lr=lrG, betas=(0.5, 0.999))
        self.optimD = optim.Adam(self.netD.parameters(), lr=lrD, betas=(0.5, 0.999))
    
    # ---------------------------
    # Losses: hinge GAN + L1 + feature-matching
    # ---------------------------
    def d_hinge_loss(self, real_pred, fake_pred):
        # real_pred, fake_pred: discriminator outputs for real and fake
        real_loss = torch.mean(F.relu(1.0 - real_pred))
        fake_loss = torch.mean(F.relu(1.0 + fake_pred))
        return 0.5 * (real_loss + fake_loss)

    def g_hinge_loss(self, fake_pred):
        # generator wants fake_pred to be large
        return -torch.mean(fake_pred)

    # Feature-matching: compute L1 between intermediate features (if you extract them)
    def feature_matching_loss(self, real_feats, fake_feats):
        loss = 0.0
        for r, f in zip(real_feats, fake_feats):
            loss = loss + F.l1_loss(f, r)
        return loss

    def train_step(self, batch, netG, netD, optimG, optimD, device,
                    l1_weight=10.0, fm_weight=5.0):
        """
        batch should contain:
        - frame1: (B,1,H,W)
        - frame2: (B,1,H,W)
        - target: (B,1,H,W) ground truth middle frame
        """

        frame1 = batch[:, 0].to(device)
        frame2 = batch[:, 2].to(device)
        target = batch[:, 1].to(device)

        # --------------------
        # 1) Update D
        # --------------------
        netD.train()
        netG.eval()
        with torch.no_grad():
            fake = netG(frame1, frame2)  # (B,1,H,W)

        # Discriminator outputs (patch maps)
        real_pred = netD(frame1, frame2, target)
        fake_pred = netD(frame1, frame2, fake.detach())

        # Hinge loss
        loss_D = self.d_hinge_loss(real_pred, fake_pred)

        optimD.zero_grad()
        loss_D.backward()
        optimD.step()

        # --------------------
        # 2) Update G
        # --------------------
        netG.train()
        netD.eval()

        fake = netG(frame1, frame2)
        fake_pred_for_g = netD(frame1, frame2, fake)

        # Adversarial loss (hinge)
        loss_g_adv = self.g_hinge_loss(fake_pred_for_g)

        # L1 reconstruction loss
        loss_l1 = F.l1_loss(fake, target)

        # (Optional) Feature matching: if netD is modified to return intermediate features,
        # you can compute fm loss here. For the simplicity of this snippet we skip extracting features.
        loss_fm = 0.0

        # Total generator loss
        loss_G = loss_g_adv + l1_weight * loss_l1 + fm_weight * loss_fm

        optimG.zero_grad()
        loss_G.backward()
        optimG.step()

        return {
            'loss_D': loss_D.item(),
            'loss_G': loss_G.item(),
            'loss_g_adv': loss_g_adv.item(),
            'loss_l1': loss_l1.item()
        }

    def train_gan(self, train_loader, val_loader, device,
                    num_epochs=50, checkpoint_dir="checkpoints_gan",
                    l1_weight=10.0, fm_weight=5.0, log_interval=50):

        os.makedirs(checkpoint_dir, exist_ok=True)

        history = {
            "train_D": [], "train_G": [],
            "train_adv": [], "train_l1": [],
            "val_l1": []
        }

        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            self.netG.train()
            self.netD.train()

            epoch_loss_D = 0.0
            epoch_loss_G = 0.0
            epoch_loss_l1 = 0.0
            epoch_loss_adv = 0.0

            for batch_idx, batch in enumerate(train_loader):
                losses = self.train_step(
                    batch, self.netG, self.netD, self.optimG, self.optimD, device,
                    l1_weight=l1_weight, fm_weight=fm_weight
                )

                epoch_loss_D += losses['loss_D']
                epoch_loss_G += losses['loss_G']
                epoch_loss_l1 += losses['loss_l1']
                epoch_loss_adv += losses['loss_g_adv']

                if batch_idx % log_interval == 0:
                    logging.info(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"D: {losses['loss_D']:.4f} | "
                        f"G: {losses['loss_G']:.4f} | "
                        f"adv: {losses['loss_g_adv']:.4f} | "
                        f"l1: {losses['loss_l1']:.4f}"
                    )

            # Average over epoch
            n_batches = len(train_loader)
            history["train_D"].append(epoch_loss_D / n_batches)
            history["train_G"].append(epoch_loss_G / n_batches)
            history["train_l1"].append(epoch_loss_l1 / n_batches)
            history["train_adv"].append(epoch_loss_adv / n_batches)

            # -------------------
            # Validation
            # -------------------
            self.netG.eval()
            val_loss_l1 = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    frame1 = batch[:, 0].to(device)
                    frame2 = batch[:, 2].to(device)
                    target = batch[:, 1].to(device)

                    fake = self.netG(frame1, frame2)
                    val_loss_l1 += F.l1_loss(fake, target).item()

            val_loss_l1 /= len(val_loader)
            history["val_l1"].append(val_loss_l1)

            logging.info(
                f"Epoch {epoch+1}/{num_epochs} finished | "
                f"D: {history['train_D'][-1]:.4f} | "
                f"G: {history['train_G'][-1]:.4f} | "
                f"Val L1: {val_loss_l1:.4f}"
            )

            # Save best model
            if val_loss_l1 < best_val_loss:
                best_val_loss = val_loss_l1
                torch.save({
                    "netG": self.netG.state_dict(),
                    "netD": self.netD.state_dict(),
                    "optimG": self.optimG.state_dict(),
                    "optimD": self.optimD.state_dict(),
                    "epoch": epoch
                }, os.path.join(checkpoint_dir, "best_model.pt"))
                logging.info(f"✅ Saved best model at epoch {epoch+1}")

        # -------------------
        # Plot losses
        # -------------------
        plt.figure(figsize=(12, 6))
        plt.plot(history["train_D"], label="Train D")
        plt.plot(history["train_G"], label="Train G")
        plt.plot(history["train_adv"], label="Train Adv")
        plt.plot(history["train_l1"], label="Train L1")
        plt.plot(history["val_l1"], label="Val L1")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("GAN Training Losses")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(checkpoint_dir, "loss_curve.png"))
        plt.close()

        return history