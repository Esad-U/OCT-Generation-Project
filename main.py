import os
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

from models import InterpolationUNet, ComplexUNetLarge, DiffusionInterpolator, UNetUpsample
from tsgan import GeneratorUNet, DiscriminatorResNet, train_tsgan
from octgan import OCTGAN
from data import ComplexFourierDataset, RegularDataset, EfficientDataset
from losses import separate_loss, combined_loss, interpolation_loss, ssim, gradient_loss, gradient_ssim_loss, PerceptualLoss, film_loss
from train import train, train_interpolation, train_diffusion, efficient_train
from visualize import plot_losses_gan, visualize_dataset_sample, visualize_model_predictions, plot_losses, evaluate_model_predictions
from evaluation import Evaluator

DATASET_PATH = '/home/esad-ugur/Data/OCT'

def fft_visualize():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = ComplexFourierDataset(
        root_dir='/mnt/storage1/esad/data/OCT/test',  # Update with your data path
        image_size=128
    )
    dataset_regular = RegularDataset(
        root_dir='/mnt/storage1/esad/data/OCT/test',
        image_size=128
    )

    a = dataset_regular[0][0][0].unsqueeze(0).unsqueeze(0).to(device)

    model = OCTGAN(hidden_channels_g=64, hidden_channels_d=64, device=device)

    a_fft = model.fft_discriminator.fft_transform(a)

    print(a_fft[0][1])
    
    # Visualize a few dataset samples
    for i in range(3):  # Visualize first 3 samples
        visualize_dataset_sample(dataset, 'lele', sample_idx=i, 
                                    save_path=f'visualizations/sample_{i}')

def vis_main(method):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    if method == 'interpolation' or method == 'gan':
        dataset = RegularDataset(
            root_dir='/home/esad-ugur/Data/OCT/test',
            image_size=256
        )
    else:
        dataset = ComplexFourierDataset(
            root_dir='/storage/esad/data/OCT/test',  # Update with your data path
            image_size=128
        )
    
    # Visualize a few dataset samples
    for i in range(3):  # Visualize first 3 samples
        visualize_dataset_sample(dataset, method, sample_idx=i, 
                                    save_path=f'visualizations/sample_{i}')
    
    # Load trained model (if available)
    if method == 'unet':
        model = ComplexUNetLarge(
            input_channels=1,
            condition_channels=2,
            hidden_channels=64,
            time_embed_dim=32
        ).to(device)
    elif method == 'interpolation':
        # model = InterpolationUNet(
        #     input_channels=1,
        #     hidden_channels=64
        # ).to(device)
        model = UNetUpsample(
            input_channels=1,
            hidden_channels=48
        ).to(device)
    elif method == 'diffusion':
        model = DiffusionInterpolator(
            input_channels=1,
            hidden_channels=64
        ).to(device)
    elif method == 'gan':
        model = OCTGAN(dataset=dataset, hidden_channels_g=48, hidden_channels_d=48, device=device)

    # Try to load the latest checkpoint
    checkpoint_dir = 'checkpoints/checkpoints_20250720_164057'
    # checkpoint_dir = 'checkpoints/best-model'
    if os.path.exists(checkpoint_dir):
        evaluator = Evaluator()
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
        checkpoint = [c for c in checkpoints if '100' in c][0]
        # print checkpoint properties

        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoint)
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            if method == 'gan':
                model.generator.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint: {latest_checkpoint}")
            
            # Visualize model predictions
            for i in range(10):  # Visualize predictions for first 3 samples
                visualize_model_predictions(model, evaluator, dataset, device, method, sample_idx=i)
            
            evaluate_model_predictions(model, evaluator, dataset, device, method)

def main(method, loss_name, optimizer_choice):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-5
    IMAGE_SIZE = 256
    HIDDEN_CHANNELS = 48
    TIME_EMBED_DIM = 32
    CHECKPOINT_FREQ = 10
    
    # Setup data
    if method == 'interpolation' or method == 'gan' or method == 'tsgan':
        # train_dataset = RegularDataset(
        #     root_dir= DATASET_PATH + '/train',
        #     image_size=IMAGE_SIZE
        # )
        # test_dataset = RegularDataset(
        #     root_dir= DATASET_PATH + '/test',
        #     image_size=IMAGE_SIZE
        # )
        train_dataset = EfficientDataset(
            image_dir=DATASET_PATH + '/train_all',
            image_size=IMAGE_SIZE,
            window_size=3
        )
        test_dataset = EfficientDataset(
            image_dir=DATASET_PATH + '/validation_all',
            image_size=IMAGE_SIZE,
            window_size=3
        )
    else:
        train_dataset = ComplexFourierDataset(
            root_dir='/storage/esad/data/OCT/train',
            image_size=IMAGE_SIZE
        )
        test_dataset = ComplexFourierDataset(
            root_dir='/storage/esad/data/OCT/test',
            image_size=IMAGE_SIZE
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    if method == 'interpolation':
        # Direct Interpolation
        # model = InterpolationUNet(
        #     input_channels=1,
        #     hidden_channels=HIDDEN_CHANNELS
        # ).to(device)
        model = UNetUpsample(
            input_channels=1,
            hidden_channels=HIDDEN_CHANNELS
        ).to(device)
    elif method == 'diffusion':
        model = DiffusionInterpolator(
            input_channels=1,
            hidden_channels=HIDDEN_CHANNELS
        ).to(device)
    elif method == 'gan':
        model = OCTGAN(
            dataset=train_dataset,
            hidden_channels_g=48,
            hidden_channels_d=48,
            device=device
        )

    # Setup optimizer
    if optimizer_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_choice == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    else:
        pass

    if loss_name == 'combined':
        loss = combined_loss
    elif loss_name == 'separate':
        loss = separate_loss
    elif loss_name == 'interpolation':
        loss = interpolation_loss
    elif loss_name == 'ssim':
        loss = ssim
    elif loss_name == 'ssim_mse':
        loss = ssim_mse
    elif loss_name == 'gradient':
        loss = gradient_loss
    elif loss_name == 'psnr':
        loss = psnr
    elif loss_name == 'gradient_ssim':
        loss = gradient_ssim_loss
    elif loss_name == 'perceptual':
        loss = PerceptualLoss(device=device)
    elif loss_name == 'film':
        loss = film_loss
    else:
        pass
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = f'checkpoints/checkpoints_{timestamp}'

    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        filemode='w',
        filename='training.log',
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    
    # Start training
    if method == 'gan':
        logging.info(f"Starting training...\nMethod: OCTGAN")
    else:
        logging.info(f"Starting training...\nMethod: {method} Upsample\nLoss: {loss_name}\nImage Size: {IMAGE_SIZE} \
                        \nOptimizer: {optimizer_choice}\nBatch Size: {BATCH_SIZE}\nHidden Channels:{HIDDEN_CHANNELS} \
                        \nLearning Rate: {LEARNING_RATE}\nDevice: {device}")

    if method == 'unet':
        train(model, train_loader, optimizer, loss, device, NUM_EPOCHS, CHECKPOINT_FREQ, checkpoint_dir=checkpoint_dir)
    elif method == 'interpolation':
        train_loss, test_loss = efficient_train(model, train_loader, test_loader, optimizer, loss, device, NUM_EPOCHS, 
                                                            CHECKPOINT_FREQ, checkpoint_dir=checkpoint_dir)
    elif method == 'diffusion':
        train_diffusion(model, train_loader, optimizer, loss, device, NUM_EPOCHS, CHECKPOINT_FREQ, checkpoint_dir=checkpoint_dir)
    elif method == 'gan':
        # g, d, fd = model.train(train_loader, NUM_EPOCHS, CHECKPOINT_FREQ, BATCH_SIZE, checkpoint_dir=checkpoint_dir)
        g, d, td = model.train_no_fft(train_loader, NUM_EPOCHS, CHECKPOINT_FREQ, BATCH_SIZE, checkpoint_dir=checkpoint_dir)
    elif method == 'tsgan':
        train_tsgan(generator1, discriminator1, generator2, discriminator2, train_loader, CHECKPOINT_FREQ, checkpoint_dir, NUM_EPOCHS)

    # Save final model
    if method == 'gan':
        final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
        torch.save(model.generator.state_dict(), final_model_path)
        logging.info(f"Training complete. Final model saved to {final_model_path}")
    else:
        final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Training complete. Final model saved to {final_model_path}")

    if method == 'interpolation':
        plot_losses(train_loss, test_loss, save_path=checkpoint_dir+'/loss.png')
        logging.info(f"Loss plot is saved.")
    elif method == 'gan':
        plot_losses_gan(g, d, td, save_path=checkpoint_dir+'/loss.png')
        logging.info(f"Loss plot is saved")

if __name__ == '__main__':
    # vis_main('interpolation')
    main('interpolation', 'film', 'adam')
    # fft_visualize()