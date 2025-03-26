import os
import random
import numpy as np
import torch
from loguru import logger
from dataset import get_voc_dataloader
from augmentation import get_transform

def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Set random seed to {seed}")

def create_data_splits(data_root, batch_size=2, num_workers=4, max_samples=None):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_root (str): Path to the data directory
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        max_samples (int, optional): Maximum number of samples to use per split.
                                    If None, use all available samples.
        
    Returns:
        dict: Dictionary containing train, val, and test dataloaders
    """
    set_seed()
    
    # Create transforms for each split
    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)
    test_transform = get_transform(train=False)
    
    # Create dataloaders with optional sample limiting
    train_loader = get_voc_dataloader(
        root=data_root,
        year='2012',
        image_set='train',
        batch_size=batch_size,
        transform=train_transform,
        num_workers=num_workers,
        shuffle=True,
        max_samples=max_samples
    )
    
    val_loader = get_voc_dataloader(
        root=data_root,
        year='2012',
        image_set='val',
        batch_size=batch_size,
        transform=val_transform,
        num_workers=num_workers,
        shuffle=False,
        max_samples=max_samples
    )
    
    # For test set, we'll use the validation set of VOC
    # In a real scenario, you might want to use a separate test set
    test_loader = get_voc_dataloader(
        root=data_root,
        year='2012',
        image_set='val',
        batch_size=batch_size,
        transform=test_transform,
        num_workers=num_workers,
        shuffle=False,
        max_samples=max_samples
    )
    
    logger.info(f"Created data splits with {len(train_loader)} training batches, "
                f"{len(val_loader)} validation batches, and {len(test_loader)} test batches")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    } 