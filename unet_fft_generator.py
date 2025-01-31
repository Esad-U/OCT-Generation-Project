import os
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

from models import InterpolationUNet, ComplexUNetLarge, DiffusionInterpolator
from data import ComplexFourierDataset, RegularDataset
from losses import separate_loss, combined_loss, interpolation_loss
from train import train, train_interpolation, train_diffusion
from visualize import visualize_dataset_sample, visualize_model_predictions

def vis_main(method):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    if method == 'interpolation':
        dataset = RegularDataset(
            root_dir='/storage/esad/data/OCT/test',
            image_size=128
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
        model = InterpolationUNet(
            input_channels=1,
            hidden_channels=96
        ).to(device)
    elif method == 'diffusion':
        model = DiffusionInterpolator(
            input_channels=1,
            hidden_channels=64
        ).to(device)

    # Try to load the latest checkpoint
    checkpoint_dir = 'checkpoints/checkpoints_20250127_213259'
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[19])
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint: {latest_checkpoint}")
            
            # Visualize model predictions
            for i in range(3):  # Visualize predictions for first 3 samples
                visualize_model_predictions(model, dataset, device, method, sample_idx=i)

def main(method, loss_name):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 500
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 256
    HIDDEN_CHANNELS = 48
    TIME_EMBED_DIM = 32
    CHECKPOINT_FREQ = 25
    
    # Setup data
    if method == 'interpolation':
        dataset = RegularDataset(
            root_dir='/storage/esad/data/OCT/train',
            image_size=IMAGE_SIZE
        )
    else:
        dataset = ComplexFourierDataset(
            root_dir='/storage/esad/data/OCT/train',
            image_size=IMAGE_SIZE
        )

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    if method == 'unet':
        model = ComplexUNetLarge(
            input_channels=1,
            condition_channels=2,
            hidden_channels=HIDDEN_CHANNELS,
            time_embed_dim=TIME_EMBED_DIM
        ).to(device)
    elif method == 'interpolation':
        # Direct Interpolation
        model = InterpolationUNet(
            input_channels=1,
            hidden_channels=HIDDEN_CHANNELS
        ).to(device)
    elif method == 'diffusion':
        model = DiffusionInterpolator(
            input_channels=1,
            hidden_channels=HIDDEN_CHANNELS
        ).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if loss_name == 'combined':
        loss = combined_loss
    elif loss_name == 'separate':
        loss = separate_loss
    elif loss_name == 'interpolation':
        loss = interpolation_loss
    
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
    logging.info(f"Starting training...\nMethod: {method}\nLoss: {loss_name}\nImage Size: {IMAGE_SIZE} \
                    \nOptimizer: Adam\nBatch Size: {BATCH_SIZE}\nHidden Channels:{HIDDEN_CHANNELS} \
                    \nLearning Rate: {LEARNING_RATE}")
    if method == 'unet':
        train(model, train_loader, optimizer, device, NUM_EPOCHS, CHECKPOINT_FREQ, checkpoint_dir=checkpoint_dir)
    elif method == 'interpolation':
        train_interpolation(model, train_loader, optimizer, loss, device, NUM_EPOCHS, CHECKPOINT_FREQ, checkpoint_dir=checkpoint_dir)
    elif method == 'diffusion':
        train_diffusion(model, train_loader, optimizer, loss, device, NUM_EPOCHS, CHECKPOINT_FREQ, checkpoint_dir=checkpoint_dir)
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Training complete. Final model saved to {final_model_path}")

if __name__ == '__main__':
    # vis_main('interpolation')
    main('interpolation', 'interpolation')