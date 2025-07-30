import os
import logging
import argparse
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.interpolation_unet import InterpolationUNet, UNetUpsample
from models.diffusion_interpolator import DiffusionInterpolator
from models.octgan import OCTGAN
from data.fourier_dataset import ComplexFourierDataset
from data.regular_dataset import RegularDataset
from data.efficient_dataset import EfficientDataset
from losses.loss_functions import (
    separate_loss, combined_loss, interpolation_loss, ssim,
    gradient_loss, gradient_ssim_loss, PerceptualLoss, film_loss
)
from train.train import train, train_interpolation, train_diffusion, efficient_train
from evaluation.visualize import (
    plot_losses_gan, visualize_dataset_sample, visualize_model_predictions,
    plot_losses, evaluate_model_predictions
)
from evaluation.evaluation import Evaluator


DATASET_PATH = '/home/esad-ugur/Data/OCT'


def get_dataset(method, image_size):
    if method in ['interpolation', 'gan']:
        train_dataset = EfficientDataset(
            image_dir=f'{DATASET_PATH}/train_all',
            image_size=image_size,
            window_size=3
        )
        test_dataset = EfficientDataset(
            image_dir=f'{DATASET_PATH}/validation_all',
            image_size=image_size,
            window_size=3
        )
    else:
        train_dataset = ComplexFourierDataset(
            root_dir='/storage/esad/data/OCT/train',
            image_size=image_size
        )
        test_dataset = ComplexFourierDataset(
            root_dir='/storage/esad/data/OCT/test',
            image_size=image_size
        )
    return train_dataset, test_dataset


def get_model(method, hidden_channels, device, dataset=None):
    if method == 'interpolation':
        return UNetUpsample(input_channels=1, hidden_channels=hidden_channels).to(device)
    elif method == 'diffusion':
        return DiffusionInterpolator(input_channels=1, hidden_channels=hidden_channels).to(device)
    elif method == 'gan':
        return OCTGAN(dataset=dataset, hidden_channels_g=hidden_channels, device=device)
    else:
        raise ValueError(f"Unsupported method: {method}")


def get_loss_fn(loss_name, device):
    losses = {
        'combined': combined_loss,
        'separate': separate_loss,
        'interpolation': interpolation_loss,
        'ssim': ssim,
        'gradient': gradient_loss,
        'gradient_ssim': gradient_ssim_loss,
        'perceptual': PerceptualLoss(device),
        'film': film_loss
    }
    return losses.get(loss_name)


def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset = get_dataset(args.method, args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(args.method, args.hidden_channels, device, train_dataset)
    optimizer = get_optimizer(args.optimizer, model, args.lr)
    loss_fn = get_loss_fn(args.loss, device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = f'checkpoints/checkpoints_{timestamp}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, filemode='w',
                        filename=os.path.join(checkpoint_dir, 'training.log'),
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Training started with method={args.method}, loss={args.loss}, optimizer={args.optimizer}")

    if args.method == 'interpolation':
        train_loss, test_loss = efficient_train(model, train_loader, test_loader, optimizer, loss_fn,
                                                device, args.epochs, args.ckpt_freq, checkpoint_dir)
        plot_losses(train_loss, test_loss, save_path=os.path.join(checkpoint_dir, 'loss.png'))

    elif args.method == 'diffusion':
        train_diffusion(model, train_loader, optimizer, loss_fn, device,
                        args.epochs, args.ckpt_freq, checkpoint_dir)

    elif args.method == 'gan':
        g_losses, d_losses = model.train_no_fft(train_loader, args.epochs,
                                                args.ckpt_freq, args.batch_size, checkpoint_dir)
        plot_losses_gan(g_losses, d_losses, save_path=os.path.join(checkpoint_dir, 'loss.png'))

    model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    if args.method == 'gan':
        torch.save(model.generator.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

    logging.info(f"Training complete. Model saved to {model_path}")


def evaluate_main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.method in ['interpolation', 'gan']:
        dataset = RegularDataset(root_dir=f'{DATASET_PATH}/test', image_size=args.image_size)
    else:
        dataset = ComplexFourierDataset(root_dir='/storage/esad/data/OCT/test', image_size=128)

    if args.method == 'interpolation':
        model = UNetUpsample(input_channels=1, hidden_channels=args.hidden_channels).to(device)
    elif args.method == 'diffusion':
        model = DiffusionInterpolator(input_channels=1, hidden_channels=args.hidden_channels).to(device)
    elif args.method == 'gan':
        model = OCTGAN(dataset=dataset, hidden_channels_g=args.hidden_channels,
                       hidden_channels_d=args.hidden_channels, device=device)

    checkpoint_dir = args.ckpt or 'checkpoints/latest'
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1]) if checkpoint_files else None

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"No valid checkpoint found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if args.method == 'gan':
        model.generator.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint: {checkpoint_path}")

    evaluator = Evaluator()
    for i in range(3):
        visualize_dataset_sample(dataset, args.method, sample_idx=i, save_path=f'visualizations/sample_{i}')
    for i in range(10):
        visualize_model_predictions(model, evaluator, dataset, device, args.method, sample_idx=i)
    evaluate_model_predictions(model, evaluator, dataset, device, args.method)


def parse_args():
    parser = argparse.ArgumentParser(description="OCT AI Training and Evaluation")
    parser.add_argument('--method', type=str, required=True,
                        choices=['interpolation', 'gan', 'diffusion'], help="Training method")
    parser.add_argument('--loss', type=str, default='interpolation', help="Loss function name")
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer name")
    parser.add_argument('--evaluate', action='store_true', help="Run evaluation instead of training")
    parser.add_argument('--ckpt', type=str, default='', help="Checkpoint directory for evaluation")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ckpt_freq', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.evaluate:
        evaluate_main(args)
    else:
        train_main(args)
