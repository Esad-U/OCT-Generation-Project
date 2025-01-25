import torch
import torch.nn as nn
import logging
import os

def train_diffusion(model, train_loader, optimizer, loss_fn, device, num_epochs, checkpoint_freq=25, log_interval=10, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.train()
    total_steps = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (odd_frames, even_frames) in enumerate(train_loader):
            odd_frames = odd_frames.to(device)
            even_frames = even_frames.to(device)
            
            optimizer.zero_grad()
            batch_size = odd_frames.shape[0]
            
            total_loss = 0
            for t in range(even_frames.shape[1]):
                # Get surrounding frames as condition
                condition = torch.cat([odd_frames[:, t], odd_frames[:, t+1]], dim=1)
                
                # Sample timestep
                timesteps = torch.randint(0, model.timesteps, (batch_size,), device=device).long()
                
                # Add noise to target frame
                noisy_frame, noise = model.get_noisy_image(even_frames[:, t], timesteps)
                
                # Predict noise
                noise_pred = model(noisy_frame, condition, timesteps)
                
                # Calculate loss
                loss = loss_fn(noise_pred, noise)
                total_loss += loss
            
            avg_loss = total_loss / even_frames.shape[1]
            avg_loss.backward()
            optimizer.step()
            
            epoch_loss += avg_loss.item()
            total_steps += 1

            # Log progress
            if batch_idx % log_interval == 0:
                logging.info(f'Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | '
                           f'Loss: {avg_loss.item():.6f}')
        
        # Log epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f'Epoch {epoch} complete | Average Loss: {avg_epoch_loss:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            logging.info(f'Saved checkpoint to {checkpoint_path}')

def train(model, train_loader, optimizer, loss_fn, device, num_epochs, checkpoint_freq=25, log_interval=10, checkpoint_dir='checkpoints'):
    """Training loop for the Complex Fourier model"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.train()
    total_steps = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (odd_frames, even_frames) in enumerate(train_loader):
            odd_frames = odd_frames.to(device)  # (B, 10, 2, H, W)
            even_frames = even_frames.to(device)  # (B, 9, 2, H, W)
            
            batch_size = odd_frames.shape[0]
            optimizer.zero_grad()
            
            # Process each time step
            total_loss = 0
            for t in range(even_frames.shape[1]):
                # Get surrounding odd frames as condition
                if t < even_frames.shape[1] - 1:
                    condition = torch.cat([odd_frames[:, t], odd_frames[:, t+1]], dim=1)
                else:
                    condition = torch.cat([odd_frames[:, t], odd_frames[:, t+1]], dim=1)
                
                # Create time tensor (normalized to [0, 1])
                time = torch.tensor([t / even_frames.shape[1]]).to(device)
                time = time.expand(batch_size)

                # Setup 1
                # noise = even_frames[:, t]
                # Setup 2
                # noise = (odd_frames[:, t] + odd_frames[:, t+1]) / 2
                # Setup 3
                noise = torch.rand(even_frames[:, t].shape).to(device)
                
                # Generate even frame
                generated = model(noise, condition, time)
                
                # Calculate loss (MSE for both magnitude and phase)
                loss = loss_fn(generated, even_frames[:, t])
                total_loss += loss
            
            # Average loss over time steps
            avg_loss = total_loss / even_frames.shape[1]
            avg_loss.backward()
            optimizer.step()
            
            epoch_loss += avg_loss.item()
            total_steps += 1
            
            # Log progress
            if batch_idx % log_interval == 0:
                logging.info(f'Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | '
                           f'Loss: {avg_loss.item():.6f}')
        
        # Log epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f'Epoch {epoch} complete | Average Loss: {avg_epoch_loss:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            logging.info(f'Saved checkpoint to {checkpoint_path}')

# Training function for direct interpolation
def train_interpolation(model, train_loader, optimizer, loss_fn, device, num_epochs, checkpoint_freq, log_interval=10, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (odd_frames, even_frames) in enumerate(train_loader):
            odd_frames = odd_frames.to(device)
            even_frames = even_frames.to(device)
            
            optimizer.zero_grad()
            
            total_loss = 0
            for t in range(even_frames.shape[1]):
                # Get surrounding odd frames
                frame1 = odd_frames[:, t]
                frame2 = odd_frames[:, t+1]
                
                # Generate intermediate frame
                predicted = model(frame1, frame2)
                
                # Calculate loss
                loss = loss_fn(predicted, even_frames[:, t])
                total_loss += loss
            
            avg_loss = total_loss / even_frames.shape[1]
            avg_loss.backward()
            optimizer.step()
            
            epoch_loss += avg_loss.item()
            
            if batch_idx % log_interval == 0:
                logging.info(f'Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | '
                      f'Loss: {avg_loss.item():.6f}')
        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            logging.info(f'Saved checkpoint to {checkpoint_path}')

