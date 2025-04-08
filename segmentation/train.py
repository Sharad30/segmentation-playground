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

def box_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of bounding boxes.
    
    Args:
        boxes1 (Tensor): First set of boxes, shape (N, 4)
        boxes2 (Tensor): Second set of boxes, shape (M, 4)
        
    Returns:
        Tensor: IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Get coordinates of intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]
    
    # Calculate area of intersection
    wh = (rb - lt).clamp(min=0)  # width-height [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # Calculate IoU
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    return iou

class Trainer:
    def __init__(self, config):
        """
        Initialize the trainer with configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # Set up logging
        log_file = os.path.join(config['output_dir'], 'training.log')
        logger.add(log_file, rotation="10 MB")
        logger.info(f"Initializing trainer with config: {json.dumps(config, indent=2)}")
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Create data loaders
        self.dataloaders = create_data_splits(
            config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            max_samples=config.get('max_samples', None)  # Use None if not specified
        )
        
        # Create model
        self.model = get_instance_segmentation_model(
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        )
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
        
        # Create learning rate scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config['lr_step_size'],
            gamma=config['lr_gamma']
        )
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        logger.info("Trainer initialized successfully")
    
    def train_one_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        
        train_loader = self.dataloaders['train']
        epoch_start = time.time()
        
        # Add gradient clipping to prevent exploding gradients
        max_norm = 1.0
        
        # At the beginning of train_one_epoch
        scaler = torch.cuda.amp.GradScaler()
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
            try:
                # Move data to device
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Inside the training loop
                with torch.cuda.amp.autocast():
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                # Check if loss_dict is valid
                if not loss_dict or not all(torch.isfinite(loss) for loss in loss_dict.values()):
                    logger.warning(f"Skipping batch with invalid loss values: {loss_dict}")
                    continue
                    
                # Skip batch if loss is not finite
                if not torch.isfinite(losses):
                    logger.warning(f"Skipping batch with non-finite loss: {losses.item()}")
                    continue
                
                # Replace optimizer steps with:
                scaler.scale(losses).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                # Use .detach() to avoid memory leaks
                total_loss += losses.detach().item()
                
                # Clear GPU cache periodically
                if torch.cuda.is_available() and train_loader.batch_size * (train_loader.dataset.__len__() // train_loader.batch_size // 10) > 0:
                    if (train_loader.batch_sampler.sampler.num_samples // train_loader.batch_size) % 10 == 0:
                        torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.error(f"CUDA error in batch: {str(e)}")
                    # Try to recover by clearing cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        val_loader = self.dataloaders['val']
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Validation {epoch}"):
                # Move data to device
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass in evaluation mode
                outputs = self.model(images)
                
                # Calculate loss manually
                batch_loss = 0
                for output, target in zip(outputs, targets):
                    # Calculate IoU-based loss for boxes if present
                    if 'boxes' in output and len(output['boxes']) > 0 and len(target['boxes']) > 0:
                        # Get predicted and ground truth boxes
                        pred_boxes = output['boxes']
                        gt_boxes = target['boxes']
                        
                        # Calculate IoU-based loss instead of MSE
                        # This handles different numbers of boxes better
                        ious = box_iou(pred_boxes, gt_boxes)
                        
                        # For each predicted box, find the best matching ground truth box
                        max_ious, _ = ious.max(dim=1)
                        
                        # Loss is 1 - IoU (higher IoU = lower loss)
                        box_loss = (1.0 - max_ious).mean()
                        batch_loss += box_loss
                    
                    # Add loss for masks if present
                    if 'masks' in output and len(output['masks']) > 0 and 'masks' in target and len(target['masks']) > 0:
                        # Simple mask loss - just to demonstrate
                        # In a real implementation, you would use a more sophisticated mask loss
                        pred_masks = output['masks'] > 0.5  # Convert to binary
                        gt_masks = target['masks']
                        
                        # Handle different numbers of masks
                        num_masks = min(pred_masks.shape[0], gt_masks.shape[0])
                        if num_masks > 0:
                            # Calculate mask IoU or dice coefficient
                            mask_loss = 0.0
                            for i in range(num_masks):
                                pred_mask = pred_masks[i].float()
                                gt_mask = gt_masks[i].float()
                                
                                # Dice coefficient: 2 * intersection / (sum1 + sum2)
                                intersection = (pred_mask * gt_mask).sum()
                                dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-6)
                                mask_loss += (1.0 - dice)
                            
                            mask_loss /= num_masks
                            batch_loss += mask_loss
                
                # If no loss was calculated, add a small placeholder loss
                if batch_loss == 0:
                    batch_loss = torch.tensor(0.1, device=self.device)
                
                total_loss += batch_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        logger.info(f"Epoch {epoch} - Validation Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_epoch = epoch
            self.save_model(os.path.join(self.config['output_dir'], 'best_model.pth'))
            logger.info(f"New best model saved with validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    def train(self):
        """Train the model for the specified number of epochs."""
        logger.info("Starting training...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            # Train
            train_loss = self.train_one_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(epoch)
            val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config['save_freq'] == 0:
                self.save_model(os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch}.pth'))
        
        logger.info(f"Training completed. Best model at epoch {self.best_epoch} with validation loss {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_model(os.path.join(self.config['output_dir'], 'final_model.pth'))
        
        # Save training history
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }
        
        with open(os.path.join(self.config['output_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f)
        
        logger.info("Training history saved")
        
        return history

def main():
    """Main function to run the training."""
    # Define configuration
    config = {
        'data_root': './data',
        'output_dir': './output_person',
        'num_classes': 2,  # Background + person
        'batch_size': 4,  # Can use larger batch size for simpler task
        'learning_rate': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_step_size': 3,
        'lr_gamma': 0.1,
        'num_epochs': 5,  # Train for more epochs
        'save_freq': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Train model
    # trainer.train()
    
    # After quick test is successful, you can train on the full dataset
    # by setting max_samples to None and continuing from the saved checkpoint
    config['max_samples'] = None
    config['output_dir'] = './output_full'
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 