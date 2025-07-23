import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from losses import PerceptualStyleLoss, film_loss

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 8, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0] * size[1]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size[0], size[1])

class Generator(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        
        # Input is two surrounding frames concatenated
        self.input_channels = input_channels * 2  # *2 for two frames, *2 for complex input
        
        # Encoder
        self.inc = self._double_conv(self.input_channels, hidden_channels)
        self.down1 = self._down_block(hidden_channels, hidden_channels * 2)
        self.down2 = self._down_block(hidden_channels * 2, hidden_channels * 4)
        self.down3 = self._down_block(hidden_channels * 4, hidden_channels * 8)
        
        # Bridge with attention
        self.bridge = nn.Sequential(
            self._double_conv(hidden_channels * 8, hidden_channels * 8),
            SelfAttention(hidden_channels * 8)
        )
        
        # Decoder with upsampling instead of transposed convolutions
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = self._double_conv(hidden_channels * 8 + hidden_channels * 4, hidden_channels * 4)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = self._double_conv(hidden_channels * 4 + hidden_channels * 2, hidden_channels * 2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = self._double_conv(hidden_channels * 2 + hidden_channels, hidden_channels)
        
        # Output layer (2 channels for complex output)
        self.outc = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    
    def _down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self._double_conv(in_channels, out_channels)
        )
    
    def encode(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        return x1, x2, x3, x4

    def decode(self, x4, x3, x2, x1):
        x = self.up3(x4)
        x = self.conv_up3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.conv_up1(torch.cat([x, x1], dim=1))

        return x
    
    def forward(self, frame1, frame2):
        # Concatenate input frames
        x = torch.cat([frame1, frame2], dim=1)
        
        x1, x2, x3, x4 = self.encode(x)
        
        # Bridge
        x4 = self.bridge(x4)
        
        x = self.decode(x4, x3, x2, x1)
        
        return nn.Tanh()(self.outc(x).squeeze(1))

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, use_condition=True):
        super(Discriminator, self).__init__()
        self.use_condition = use_condition
        # If conditional, input is: frame_t-1 + frame_gen + frame_t+1 => 3 channels
        # If unconditional, input is only frame_gen => 1 channel
        total_in_channels = in_channels * 3 if use_condition else in_channels

        def conv_block(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            # (b, total_in_channels, 256, 256) -> (b, 64, 128, 128)
            nn.Conv2d(total_in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            conv_block(64, 128, stride=2),    # (b, 128, 64, 64)
            conv_block(128, 256, stride=2),   # (b, 256, 32, 32)
            conv_block(256, 512, stride=1),   # (b, 512, 31, 31)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # (b, 1, 30, 30)
            nn.Sigmoid()  # For BCE loss
        )

    def forward(self, frame_gen, frame_prev=None, frame_next=None):
        if self.use_condition:
            x = torch.cat([frame_prev, frame_gen, frame_next], dim=1)
        else:
            x = frame_gen
        return self.model(x)

class Discriminator3D(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=64):
        super(Discriminator3D, self).__init__()

        self.model = nn.Sequential(
            # Input: (B, 1, 3, 128, 128)
            nn.Conv3d(input_channels, hidden_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(hidden_channels, hidden_channels * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(hidden_channels * 2, hidden_channels * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(hidden_channels * 4, hidden_channels * 8, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(hidden_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Final layer to reduce to 1 score per sequence
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input x: (B, T=3, C=1, H=128, W=128)
        x = x.unsqueeze(2)
        x = x.permute(0, 2, 1, 3, 4)  # → (B, C=1, T=3, H, W)
        return self.model(x)

class PatchGanDiscriminator(nn.Module):
    def __init__(self, input_channels=3, base_channels=64):
        super().__init__()

        self.model = nn.Sequential(
            # No norm in first layer
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),  # -> H/2
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),  # -> H/4
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),  # -> H/8
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1),  # -> slightly less spatial shrinkage
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),  # Final grid of real/fake scores
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class FFTDiscriminator(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=16):
        super().__init__()
        # Double the input channels to accommodate both real and imaginary parts
        self.fft_channels = input_channels * 2
        
        self.model = nn.Sequential(
            nn.Conv2d(self.fft_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_channels * 8, hidden_channels * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool2d(1),  # Force output to be 1x1 regardless of input size
            nn.Conv2d(hidden_channels * 16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def fft_transform(self, img):
        # Compute 2D FFT
        fft = torch.fft.fft2(img, norm='ortho')
        
        # Extract real and imaginary components
        real_part = fft.real
        imag_part = fft.imag
        
        # Stack real and imaginary parts along the channel dimension
        # For each input channel, we now have two channels (real and imaginary)
        batch_size, channels, height, width = img.shape
        fft_components = torch.empty(batch_size, channels * 2, height, width, device=img.device)
        
        for c in range(channels):
            fft_components[:, c*2, :, :] = real_part[:, c, :, :]
            fft_components[:, c*2+1, :, :] = imag_part[:, c, :, :]
            
        return fft_components
    
    def forward(self, img):
        # Transform image to frequency domain representation with both real and imaginary parts
        fft_img = self.fft_transform(img)
        return self.model(fft_img).view(-1, 1)


class OCTGAN():
    def __init__(self, dataset, hidden_channels_g, device):
        self.generator = Generator(input_channels=1, hidden_channels=hidden_channels_g).to(device)

        self.discriminator = Discriminator(in_channels=1).to(device)

        # self.discriminator3d = Discriminator3D(input_channels=1, hidden_channels=hidden_channels_d).to(device)

        # self.patch_discriminator = PatchGanDiscriminator().to(device)

        # self.fft_discriminator = FFTDiscriminator(hidden_channels=hidden_channels_d).to(device)

        # self.dataset = dataset

        self.device = device

        self.psl = PerceptualStyleLoss().to(device)

    def train(self, train_loader, num_epochs, checkpoint_freq, batch_size, lr=0.0002, beta1=0.5, log_interval=10, checkpoint_dir='checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        g_losses = []
        d_losses = []
        fd_losses = []

        criterion = nn.BCELoss()
        criterion_fd = nn.BCELoss()
        # gen_loss = PerceptualLoss()
        # gen_loss = nn.L1Loss()
        gen_loss = ssim_mse

        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_fft_d = optim.Adam(self.fft_discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        # Add learning rate schedulers
        scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.95)
        scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.95)
        scheduler_f = optim.lr_scheduler.ExponentialLR(optimizer_fft_d, gamma=0.95)

        logging.info("Label smoothing is applied. FFT closed. Ssim used")
        
        for epoch in range(num_epochs):
            # Training phase
            
            # Create real and fake labels
            # real_labels_d = torch.ones(batch_size, 1, device=self.device)
            # fake_labels_d = torch.zeros(batch_size, 1, device=self.device)
            # real_labels_f = torch.ones(batch_size, 1, device=self.device)
            # fake_labels_f = torch.zeros(batch_size, 1, device=self.device)
            real_labels_d = torch.rand(batch_size, 1, device=self.device) * 0.1 + 0.9
            fake_labels_d = torch.rand(batch_size, 1, device=self.device) * 0.1
            real_labels_f = torch.rand(batch_size, 1, device=self.device) * 0.1 + 0.9
            fake_labels_f = torch.rand(batch_size, 1, device=self.device) * 0.1

            # Initialize epoch losses
            epoch_loss_g = 0
            epoch_loss_d = 0
            epoch_loss_fd = 0

            for batch_idx, (odd_frames, even_frames) in enumerate(train_loader):
                odd_frames = odd_frames.to(self.device)
                even_frames = even_frames.to(self.device)

                total_lg = 0
                total_ld = 0
                total_lfd = 0
                
                for t in range(even_frames.shape[1]):
                    pre = odd_frames[:, t].unsqueeze(1)
                    post = odd_frames[:, t+1].unsqueeze(1)
                    central = even_frames[:, t].unsqueeze(1)

                    real_combined = torch.cat([pre, central, post], dim=0)
                    real_sequence = torch.cat([pre, central, post], dim=1)

                    central_fake = self.generator(pre, post).unsqueeze(1)
                    pre_central = self.generator(pre, central).unsqueeze(1)
                    central_post = self.generator(central, post).unsqueeze(1)
                    # central_fake_2 = self.generator(pre_central, central_post)
                    fake_sequence = torch.cat([pre_central, central_fake, central_post], dim=1)

                    # Train regular discriminator
                    self.discriminator.train()

                    real_outputs = self.discriminator(central)
                    real_loss = criterion(real_outputs, real_labels_d)
                    
                    fake_outputs = self.discriminator(central_fake.detach())
                    fake_loss = criterion(fake_outputs, fake_labels_d)

                    d_loss = real_loss + fake_loss

                    optimizer_d.zero_grad()
                    d_loss.backward()
                    optimizer_d.step()
                    total_ld += d_loss.item()

                    # Train FFT discriminator
                    self.fft_discriminator.train()

                    real_outputs = self.fft_discriminator(real_sequence)
                    real_loss = criterion_fd(real_outputs, real_labels_f)

                    fake_outputs = self.fft_discriminator(fake_sequence.detach())
                    fake_loss = criterion_fd(fake_outputs, fake_labels_f)

                    fft_d_loss = real_loss + fake_loss

                    optimizer_fft_d.zero_grad()
                    fft_d_loss.backward()
                    optimizer_fft_d.step()
                    total_lfd += fft_d_loss.item()

                    # Train Generator
                    self.generator.train()
                    self.discriminator.eval()
                    self.fft_discriminator.eval()

                    ## Regenerate for generator training
                    central_fake = self.generator(pre, post).unsqueeze(1)
                    pre_central = self.generator(pre, central).unsqueeze(1)
                    central_post = self.generator(central, post).unsqueeze(1)
                    fake_sequence = torch.cat([pre_central, central_fake, central_post], dim=1)

                    fakes_loss = gen_loss(central_fake, central)
                    # fakes_loss = 0

                    fake_outputs_d = self.discriminator(central_fake)
                    fake_outputs_fft_d = self.fft_discriminator(fake_sequence)
                    g_loss = fakes_loss + criterion(fake_outputs_d, real_labels_d) + criterion_fd(fake_outputs_fft_d, real_labels_f)

                    optimizer_g.zero_grad()
                    g_loss.backward()
                    optimizer_g.step()
                    total_lg += g_loss.item()
                
                if batch_idx % log_interval == 0:
                    logging.info(f'Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | '
                                    f'Generator Loss: {total_lg:.4f} - Discriminator Loss: {total_ld:.4f} - FFT Discriminator Loss: {total_lfd:.4f}')

                epoch_loss_g += total_lg
                epoch_loss_d += total_ld
                epoch_loss_fd += total_lfd

            epoch_loss_g /= len(train_loader)
            epoch_loss_d /= len(train_loader)
            epoch_loss_fd /= len(train_loader)

            g_losses.append(epoch_loss_g)
            d_losses.append(epoch_loss_d)
            fd_losses.append(epoch_loss_fd)

            scheduler_g.step()
            scheduler_d.step()
            scheduler_f.step()

            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'gen_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': optimizer_g.state_dict(),
                }, checkpoint_path)
                logging.info(f'Saved checkpoint to {checkpoint_path}')

        return g_losses, d_losses, fd_losses

    def train_no_fft(self, train_loader, num_epochs, checkpoint_freq, batch_size, lr=2e-4, beta1=0.5, log_interval=10, checkpoint_dir='checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        g_losses = []
        d_losses = []
        # td_losses = []

        criterion = nn.BCELoss()
        # gen_loss = PerceptualLoss()
        # gen_loss = nn.L1Loss()
        gen_loss = film_loss

        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr/2, betas=(beta1, 0.999))
        # optimizer_td = optim.Adam(self.discriminator3d.parameters(), lr=lr, betas=(beta1, 0.999))

        # Add learning rate schedulers
        # scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.95)
        # scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.95)
        # scheduler_td = optim.lr_scheduler.ExponentialLR(optimizer_td, gamma=0.95)

        for epoch in range(num_epochs):
            # Training phase
            
            # Create real and fake labels
            # real_labels_d = torch.ones(batch_size, 1, device=self.device)
            # fake_labels_d = torch.zeros(batch_size, 1, device=self.device)
            # real_labels_d = torch.rand(batch_size, 1, device=self.device) * 0.05 + 0.95
            # fake_labels_d = torch.rand(batch_size, 1, device=self.device) * 0.05

            # Initialize epoch losses
            epoch_loss_g = 0
            epoch_loss_d = 0

            for batch_idx, sequence in enumerate(train_loader):
                sequence = sequence.to(self.device)

                pre = sequence[:, 0].unsqueeze(1) # (B, 1, H, W)
                central = sequence[:, 1].unsqueeze(1)
                post = sequence[:, 2].unsqueeze(1)

                # Might not use this
                real_sequence = torch.cat([pre, central, post], dim=1) # (B, 3, H, W)

                self.generator.eval()
                # Generate the fake image(s) using the model
                with torch.no_grad():
                    central_fake = self.generator(pre, post).unsqueeze(1)
                    # pre_central = self.generator(pre, central).unsqueeze(1)
                    # central_post = self.generator(central, post).unsqueeze(1)
                    # central_fake_2 = self.generator(pre_central, central_post).unsqueeze(1)
                # fake_sequence = torch.cat([pre_central, central_fake_2, central_post], dim=1)

                # Train discriminator
                self.discriminator.train()
                # real_sample = self.dataset.sample_random_images(batch_size).to(self.device)
                real_outputs = self.discriminator(central, pre, post)
                real_labels_d = torch.ones_like(real_outputs)
                # real_labels_d = torch.rand(real_outputs.shape, device=self.device) * 0.05 + 0.95
                real_loss = criterion(real_outputs, real_labels_d)

                fake_outputs = self.discriminator(central_fake.detach(), pre, post)

                fake_labels_d = torch.zeros_like(fake_outputs)
                # fake_labels_d = torch.rand(fake_outputs.shape, device=self.device) * 0.05
                fake_loss = criterion(fake_outputs, fake_labels_d)

                d_loss = (real_loss + fake_loss) * 0.5

                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()
                # Train discriminator ends

                # Train Generator
                self.generator.train()
                self.discriminator.eval()
                if epoch <= num_epochs/2:
                    wts = [1, 1, 1]
                else:
                    wts = [1, 0.25, 40]

                ## Regenerate for generator training
                central_fake = self.generator(pre, post).unsqueeze(1)
                # pre_central = self.generator(pre, central).unsqueeze(1)
                # central_post = self.generator(central, post).unsqueeze(1)
                # central_fake_2 = self.generator(pre_central, central_post).unsqueeze(1)
                # fake_sequence = torch.cat([pre_central, central_fake_2, central_post], dim=1)

                loss_G_film = gen_loss(central_fake, central, self.psl, wts)

                fake_outputs = self.discriminator(central_fake, pre, post)
                # real_labels = torch.rand(fake_outputs_d.shape, device=self.device) * 0.05 + 0.95
                real_labels = torch.ones_like(fake_outputs)
                loss_G_adv = criterion(fake_outputs, real_labels)

                # REORGANIZE THIS LOSS !!!
                g_loss = loss_G_adv * 0.1 + loss_G_film * 0.9

                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()
                    
                if batch_idx % log_interval == 0:
                    logging.info(f'Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | '
                                    f'Generator Loss: {g_loss.item():.4f} - Discriminator Loss: {d_loss.item():.4f}')

                epoch_loss_g += g_loss.item()
                epoch_loss_d += d_loss.item()

            epoch_loss_g /= len(train_loader)
            epoch_loss_d /= len(train_loader)

            g_losses.append(epoch_loss_g)
            d_losses.append(epoch_loss_d)

            # scheduler_g.step()
            # scheduler_d.step()

            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'gen_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': optimizer_g.state_dict(),
                }, checkpoint_path)
                logging.info(f'Saved checkpoint to {checkpoint_path}')

        return g_losses, d_losses