import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np

# TODO: Try the model

# Define a deeper U-Net Generator with residual connections
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.conv(x)

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(GeneratorUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(256)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, img1, img2):
        x = torch.cat((img1, img2), dim=1)
        x = self.encoder(x)
        return self.decoder(x)

# Define a deeper Discriminator with spectral normalization
class DiscriminatorResNet(nn.Module):
    def __init__(self, in_channels=1):
        super(DiscriminatorResNet, self).__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(),
            ResidualBlock(64),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(),
            ResidualBlock(128),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1)
        )
    
    def forward(self, img):
        return self.model(img)

# WGAN with Gradient Penalty
class WGAN_GP(nn.Module):
    def __init__(self, generator, discriminator, lr=1e-4, lambda_gp=10):
        super(WGAN_GP, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_gp = lambda_gp
        self.opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(d_interpolates),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return self.lambda_gp * penalty
    
    def train_discriminator(self, real_data, fake_data):
        d_real = self.discriminator(real_data)
        d_fake = self.discriminator(fake_data.detach())
        gp = self.compute_gradient_penalty(real_data, fake_data)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake) + gp
        self.opt_d.zero_grad()
        d_loss.backward()
        self.opt_d.step()
        return d_loss.item()
    
    def train_generator(self, fake_data):
        g_loss = -torch.mean(self.discriminator(fake_data))
        self.opt_g.zero_grad()
        g_loss.backward()
        self.opt_g.step()
        return g_loss.item()

def train_tsgan(generator1, discriminator1, generator2, discriminator2, dataloader, checkpoint_freq, checkpoint_dir, epochs=100, lambda_gp=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator1.to(device)
    discriminator1.to(device)
    generator2.to(device)
    discriminator2.to(device)
    
    wgan1 = WGAN_GP(generator1, discriminator1, lr=1e-4, lambda_gp=lambda_gp)
    wgan2 = WGAN_GP(generator2, discriminator2, lr=1e-4, lambda_gp=lambda_gp)
    
    for epoch in range(epochs):
        print("aaaaa")
        for batch_idx, (odd_frames, even_frames) in enumerate(dataloader):
            odd_frames = odd_frames.to(device)
            even_frames = even_frames.to(device)

            for t in range(even_frames.shape[1]):
                img1 = odd_frames[:, t].unsqueeze(1)
                img2 = odd_frames[:, t+1].unsqueeze(1)
                print(img1.shape)
                print(img2.shape)
                real_mid = even_frames[:, t].unsqueeze(1)
            
                # Train WGAN1
                fake_mid = generator1(img1, img2)
                d_loss1 = wgan1.train_discriminator(real_mid, fake_mid)
                g_loss1 = wgan1.train_generator(fake_mid)
                
                # Train WGAN2
                fake_slice = generator2(fake_mid)
                d_loss2 = wgan2.train_discriminator(real_mid, fake_slice)
                g_loss2 = wgan2.train_generator(fake_slice)
            
        print(f"Epoch [{epoch+1}/{epochs}] | D1 Loss: {d_loss1:.4f} | G1 Loss: {g_loss1:.4f} | D2 Loss: {d_loss2:.4f} | G2 Loss: {g_loss2:.4f}")

        if epoch % checkpoint_freq == 0:
            # Save model checkpoints
            torch.save(generator1.state_dict(), f"{checkpoint_dir}/generator1_epoch{epoch+1}.pth")
            torch.save(discriminator1.state_dict(), f"{checkpoint_dir}/discriminator1_epoch{epoch+1}.pth")
            torch.save(generator2.state_dict(), f"{checkpoint_dir}/generator2_epoch{epoch+1}.pth")
            torch.save(discriminator2.state_dict(), f"{checkpoint_dir}/discriminator2_epoch{epoch+1}.pth")
