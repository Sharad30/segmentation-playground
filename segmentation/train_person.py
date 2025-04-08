import os
import time
import datetime
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from loguru import logger
from tqdm import tqdm
import json

from data_splits import create_data_splits
from model import get_instance_segmentation_model

class Trainer:
    def __init__(self, config):
        """
        Initialize the trainer with the given configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Create data loaders
        self.dataloaders = create_data_splits(
            config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            max_samples=config.get('max_samples')
        )
        
        # Create model
        self.model = get_instance_segmentation_model(num_classes=config['num_classes'])
        self.model.to(self.device)
        
        # Create optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(
            params, 
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
        
        # Create learning rate scheduler
        self.lr_scheduler = StepLR(
            self.optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
        # Initialize best model info
        self.best_val_loss = float('inf')
        
        logger.info(f"Initialized trainer with config: {config}")
        
    def train(self):
        """
        Train the model for the specified number of epochs.
        """
        logger.info(f"Starting training for {self.config['num_epochs']} epochs")
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            epoch_start = time.time()
            logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train for one epoch
            train_loss = self._train_one_epoch(epoch)
            
            # Evaluate on validation set
            val_loss = self._validate(epoch)
            
            # Update learning rate
            self.lr_scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.lr_scheduler.get_last_lr()[0])
            
            epoch_time = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_time)
            
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"LR: {self.lr_scheduler.get_last_lr()[0]:.6f}")
            
            # Save model if it's the best so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model(os.path.join(self.config['output_dir'], 'best_model.pth'))
                logger.info(f"Saved new best model with validation loss: {val_loss:.4f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config['save_freq'] == 0:
                self._save_model(os.path.join(self.config['output_dir'], f'model_epoch_{epoch+1}.pth'))
                logger.info(f"Saved checkpoint at epoch {epoch+1}")
            
            # Save training history
            self._save_history()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {datetime.timedelta(seconds=int(total_time))}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
    def _train_one_epoch(self, epoch):
        """
        Train the model for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            float: Average loss for this epoch
        """
        self.model.train()
        total_loss = 0
        
        # Get train dataloader
        train_loader = self.dataloaders['train']
        
        # Use tqdm for progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", unit="batch") as pbar:
            for images, targets in pbar:
                # Move data to device
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                loss_dict = self.model(images, targets)
                
                # Handle different return types
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                elif isinstance(loss_dict, list):
                    losses = sum(loss for loss in loss_dict if isinstance(loss, (int, float, torch.Tensor)))
                else:
                    losses = loss_dict
                
                # Backward pass and optimize
                losses.backward()
                self.optimizer.step()
                
                # Update statistics
                total_loss += losses.item()
                pbar.set_postfix(loss=losses.item())
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss
    
    def _validate(self, epoch):
        """
        Validate the model on the validation set.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        # Get validation dataloader
        val_loader = self.dataloaders['val']
        
        # No gradient computation for validation
        with torch.no_grad():
            # Use tqdm for progress bar
            with tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", unit="batch") as pbar:
                for images, targets in pbar:
                    # Move data to device
                    images = [image.to(self.device) for image in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # Forward pass - explicitly pass targets to get losses even in eval mode
                    outputs = self.model(images, targets)
                    
                    # Handle different return types
                    if isinstance(outputs, dict):
                        losses = sum(loss for loss in outputs.values())
                    elif isinstance(outputs, list):
                        # If we got predictions instead of losses, we need to compute the loss manually
                        # This is a simplified approach - you might need to implement proper loss calculation
                        logger.warning("Model returned predictions instead of losses during validation")
                        losses = torch.tensor(0.0, device=self.device)  # Placeholder
                    else:
                        losses = outputs
                    
                    # Update statistics
                    total_loss += losses.item()
                    pbar.set_postfix(loss=losses.item())
        
        # Calculate average loss
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss
    
    def _save_model(self, path):
        """
        Save the model checkpoint.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }, path)
    
    def _save_history(self):
        """
        Save the training history to a JSON file.
        """
        history_path = os.path.join(self.config['output_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

def main():
    # Define configuration
    config = {
        'data_root': './data',
        'output_dir': './output_person',
        'num_classes': 2,  # Background + person
        'batch_size': 4,  # Can use larger batch size for simpler task
        'num_workers': 2,
        'learning_rate': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_step_size': 3,
        'lr_gamma': 0.1,
        'num_epochs': 1,  # Train for more epochs
        'save_freq': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main() 