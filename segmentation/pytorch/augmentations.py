import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np
from loguru import logger

class Compose:
    """Compose several transforms together"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    """Resize image and mask to a specified size"""
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image, mask):
        # Resize image
        image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
        # Resize mask (nearest neighbor to avoid creating new classes)
        mask = F.resize(mask.unsqueeze(0), self.size, interpolation=F.InterpolationMode.NEAREST).squeeze(0)
        return image, mask


class RandomResize:
    """Randomly resize with a factor between min_scale and max_scale"""
    def __init__(self, min_scale=0.5, max_scale=2.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, mask):
        scale = random.uniform(self.min_scale, self.max_scale)
        h, w = image.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = F.resize(image, (new_h, new_w), interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resize(mask.unsqueeze(0), (new_h, new_w), interpolation=F.InterpolationMode.NEAREST).squeeze(0)
        return image, mask


class RandomCrop:
    """Randomly crop image and mask to a specified size"""
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image, mask):
        # Get dimensions
        _, h, w = image.shape
        th, tw = self.size
        
        # Ensure crop size is not larger than image
        if h < th or w < tw:
            # Resize to at least the crop size
            scale = max(th / h, tw / w) * 1.1  # Add 10% margin
            new_h, new_w = int(h * scale), int(w * scale)
            image = F.resize(image, (new_h, new_w), interpolation=F.InterpolationMode.BILINEAR)
            mask = F.resize(mask.unsqueeze(0), (new_h, new_w), interpolation=F.InterpolationMode.NEAREST).squeeze(0)
            h, w = new_h, new_w
        
        # Random crop
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        image = image[:, i:i+th, j:j+tw]
        mask = mask[i:i+th, j:j+tw]
        
        return image, mask


class RandomHorizontalFlip:
    """Randomly flip image and mask horizontally"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    """Randomly flip image and mask vertically"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask


class ColorJitter:
    """Apply color jitter to image (mask is unchanged)"""
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, mask):
        image = self.jitter(image)
        return image, mask


class Normalize:
    """Normalize image with mean and std"""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask


class PadToSize:
    """Pad image and mask to a specified size"""
    def __init__(self, size, fill=0):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.fill = fill

    def __call__(self, image, mask):
        # Get dimensions
        _, h, w = image.shape
        th, tw = self.size
        
        # Calculate padding
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        
        if pad_h > 0 or pad_w > 0:
            padding = [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2]
            image = F.pad(image, padding, fill=self.fill)
            mask = F.pad(mask.unsqueeze(0), padding, fill=0).squeeze(0)
        
        return image, mask


def get_transform(config, is_train=True):
    """
    Create transform pipeline based on configuration
    
    Args:
        config: Configuration dictionary
        is_train: Whether to create transforms for training or evaluation
    
    Returns:
        transform: Composed transform function
    """
    input_size = config.get('input_size', 512)
    mean = config.get('mean', (0.485, 0.456, 0.406))
    std = config.get('std', (0.229, 0.224, 0.225))
    
    if is_train:
        transforms = [
            RandomResize(min_scale=0.5, max_scale=2.0),
            RandomCrop(input_size),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(
                brightness=config.get('brightness', 0.1),
                contrast=config.get('contrast', 0.1),
                saturation=config.get('saturation', 0.1),
                hue=config.get('hue', 0.1)
            ),
            Normalize(mean=mean, std=std)
        ]
        logger.info(f"Created training transforms with input size {input_size}")
    else:
        transforms = [
            Resize(input_size),
            Normalize(mean=mean, std=std)
        ]
        logger.info(f"Created validation transforms with input size {input_size}")
    
    return Compose(transforms) 