import os
import time
import argparse
import yaml
import random
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from loguru import logger

from segmentation.pytorch.dataset import get_data_loaders
from segmentation.pytorch.model import create_model

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def setup_logger(log_dir):
    """Setup loguru logger"""
    os.makedirs(log_dir, exist_ok=True)
    logger.add(
        os.path.join(log_dir, "train_{time}.log"),
        rotation="100 MB",
        retention="30 days",
        level="INFO"
    )
    logger.info(f"Logger setup complete. Logs will be saved to {log_dir}")

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Dice coefficient for segmentation.
    
    Args:
        pred: predicted masks, shape [B, C, H, W]
        target: target masks, shape [B, H, W]
        smooth: smoothing factor to avoid division by zero
    
    Returns:
        dice coefficient per class
    """
    num_classes = pred.shape[1]
    pred = F.softmax(pred, dim=1)
    
    # One-hot encode target
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    
    # Flatten tensors
    pred_flat = pred.view(-1, num_classes, pred.size(-2) * pred.size(-1))
    target_flat = target_one_hot.view(-1, num_classes, target_one_hot.size(-2) * target_one_hot.size(-1))
    
    # Calculate per-class Dice coefficient
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.mean(dim=0)  # Average over batch, return per-class

def compute_metrics(pred, target, num_classes):
    """
    Compute various segmentation metrics.
    
    Args:
        pred: predicted masks, shape [B, C, H, W]
        target: target masks, shape [B, H, W]
        num_classes: number of classes
    
    Returns:
        dict with metrics
    """
    # Get predicted class
    pred_class = torch.argmax(pred, dim=1)
    
    # Calculate pixel accuracy
    correct = (pred_class == target).float().sum()
    total = target.numel()
    pixel_acc = correct / total
    
    # Calculate dice coefficient
    dice = dice_coefficient(pred, target)
    
    # Average dice across classes (excluding background)
    mean_dice = dice[1:].mean() if num_classes > 1 else dice[0]
    
    return {
        'pixel_acc': pixel_acc.item(),
        'mean_dice': mean_dice.item(),
        'dice_per_class': dice.tolist()
    }

def train_epoch(model, dataloader, criterion, optimizer, device, config, epoch, scaler=None):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_metrics = {
        'pixel_acc': 0,
        'mean_dice': 0
    }
    
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{config['epochs']}, Train") as pbar:
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass with mixed precision
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                # Backward and optimize with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Compute metrics
            metrics = compute_metrics(outputs, masks, config['num_classes'])
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_metrics['pixel_acc'] += metrics['pixel_acc']
            epoch_metrics['mean_dice'] += metrics['mean_dice']
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': metrics['pixel_acc'],
                'dice': metrics['mean_dice']
            })
    
    # Compute epoch averages
    epoch_loss /= len(dataloader)
    epoch_metrics['pixel_acc'] /= len(dataloader)
    epoch_metrics['mean_dice'] /= len(dataloader)
    
    return epoch_loss, epoch_metrics

def validate(model, dataloader, criterion, device, config):
    """Validate the model"""
    model.eval()
    val_loss = 0
    val_metrics = {
        'pixel_acc': 0,
        'mean_dice': 0
    }
    
    class_dice_sum = torch.zeros(config['num_classes'], device=device)
    
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Validation") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Compute metrics
                metrics = compute_metrics(outputs, masks, config['num_classes'])
                
                # Update metrics
                val_loss += loss.item()
                val_metrics['pixel_acc'] += metrics['pixel_acc']
                val_metrics['mean_dice'] += metrics['mean_dice']
                
                # Accumulate per-class dice
                class_dice_sum += torch.tensor(metrics['dice_per_class'], device=device)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': metrics['pixel_acc'],
                    'dice': metrics['mean_dice']
                })
    
    # Compute averages
    val_loss /= len(dataloader)
    val_metrics['pixel_acc'] /= len(dataloader)
    val_metrics['mean_dice'] /= len(dataloader)
    val_metrics['dice_per_class'] = (class_dice_sum / len(dataloader)).tolist()
    
    return val_loss, val_metrics

def train(config):
    """Main training function"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create run-specific directories inside artifacts/pytorch
    base_dir = Path("artifacts/pytorch")
    run_dir = base_dir / config['run_name']
    output_dir = run_dir / "output"
    log_dir = run_dir / "logs"
    
    # Create directories
    base_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger with run-specific log file
    logger.add(
        log_dir / f"train_{config['run_name']}.log",
        rotation="100 MB",
        retention="30 days",
        level="INFO"
    )
    
    # Initialize wandb
    if config['use_wandb']:
        wandb.init(
            project=config['wandb_project'],
            name=config['run_name'],
            config=config
        )
        logger.info(f"Initialized wandb with project: {config['wandb_project']}, run: {config['run_name']}")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(config)
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Create criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create learning rate scheduler if configured
    if config.get('use_lr_scheduler', False):
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )
    else:
        scheduler = None
    
    # Setup mixed precision training if available
    if config['use_amp'] and torch.cuda.is_available():
        scaler = GradScaler()
        logger.info("Using mixed precision training")
    else:
        scaler = None
        logger.info("Using full precision training")
    
    # Training loop
    best_dice = 0
    
    logger.info(f"Starting training for {config['epochs']} epochs")
    for epoch in range(config['epochs']):
        # Train for one epoch
        start_time = time.time()
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch, scaler
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, config)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = config['learning_rate']
        
        # Epoch time
        epoch_time = time.time() - start_time
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{config['epochs']}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_metrics['pixel_acc']:.4f}, "
            f"Train Dice: {train_metrics['mean_dice']:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_metrics['pixel_acc']:.4f}, "
            f"Val Dice: {val_metrics['mean_dice']:.4f}, "
            f"LR: {current_lr:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Log per-class Dice in validation
        if val_metrics.get('dice_per_class'):
            dice_per_class = val_metrics['dice_per_class']
            class_names = val_loader.dataset.class_names
            
            for i, (name, dice) in enumerate(zip(class_names, dice_per_class)):
                if i == 0 and name == 'background':
                    continue  # Skip background class in logging
                logger.info(f"Class {name}: Dice = {dice:.4f}")
        
        # Log to wandb
        if config['use_wandb']:
            wandb_log = {
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/pixel_acc': train_metrics['pixel_acc'],
                'train/mean_dice': train_metrics['mean_dice'],
                'val/loss': val_loss,
                'val/pixel_acc': val_metrics['pixel_acc'],
                'val/mean_dice': val_metrics['mean_dice'],
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }
            
            # Log per-class Dice
            if val_metrics.get('dice_per_class'):
                for i, (name, dice) in enumerate(zip(class_names, dice_per_class)):
                    wandb_log[f'val/dice_{name}'] = dice
            
            wandb.log(wandb_log)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_dice': val_metrics['mean_dice'],
            'config': config
        }
        
        # Save latest checkpoint
        torch.save(
            checkpoint,
            output_dir / f"checkpoint_latest.pth"
        )
        
        # Save best model
        if val_metrics['mean_dice'] > best_dice:
            best_dice = val_metrics['mean_dice']
            torch.save(
                checkpoint,
                output_dir / f"checkpoint_best.pth"
            )
            logger.info(f"Saved new best model with Dice: {best_dice:.4f}")
    
    # Save final metrics
    metrics = {
        'val_dice': best_dice,
        'val_pixel_acc': val_metrics['pixel_acc'],
        'val_loss': val_loss,
        'epochs': config['epochs'],
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'model': config['model'],
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR' if config.get('use_lr_scheduler', False) else 'None',
        'weight_decay': config['weight_decay'],
        'use_amp': config['use_amp']
    }
    
    # Save metrics to yaml file
    with open(output_dir / 'metrics.yaml', 'w') as f:
        yaml.dump(metrics, f)
    
    # Finish training
    logger.info(f"Training completed. Best validation Dice: {best_dice:.4f}")
    
    # Close wandb
    if config['use_wandb']:
        wandb.finish()
    
    return best_dice

def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet for semantic segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Setup logger
    setup_logger(config['log_dir'])
    
    # Log config
    logger.info(f"Configuration: {config}")
    
    # Start training
    train(config)

if __name__ == '__main__':
    main() 