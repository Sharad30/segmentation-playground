import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np
from loguru import logger
from PIL import Image

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
            
        # Final validation: ensure all boxes have positive width and height
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    # Fix the box by adding a small padding
                    x2 = max(x1 + 1, x2)
                    y2 = max(y1 + 1, y2)
                    boxes[i] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            
            target['boxes'] = boxes
            
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            
            # Flip masks
            if 'masks' in target:
                target['masks'] = torch.flip(target['masks'], [2])
            
            # Flip boxes
            if 'boxes' in target:
                # Check if image is a tensor or PIL Image
                if isinstance(image, torch.Tensor):
                    # For tensor, shape is [C, H, W]
                    _, h, w = image.shape
                else:
                    # For PIL Image
                    w, h = image.size
                
                boxes = target['boxes']
                boxes = boxes[:, [0, 1, 2, 3]].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
                
        return image, target

class RandomRotation:
    def __init__(self, degrees, prob=0.5):
        self.degrees = degrees
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            angle = random.uniform(-self.degrees, self.degrees)
            
            # Get image dimensions
            if isinstance(image, torch.Tensor):
                _, h, w = image.shape
            else:
                w, h = image.size
            
            # Calculate image center
            center = (w / 2, h / 2)
            
            # Rotate image
            image = F.rotate(image, angle)
            
            # Rotate masks if present
            if 'masks' in target:
                # Convert masks to PIL images, rotate them, and convert back to tensor
                masks = target['masks']
                rotated_masks = []
                
                for mask in masks:
                    # Convert to PIL Image
                    mask_pil = Image.fromarray(mask.cpu().numpy())
                    # Rotate mask
                    rotated_mask_pil = F.rotate(mask_pil, angle)
                    # Convert back to tensor
                    rotated_mask = torch.from_numpy(np.array(rotated_mask_pil))
                    rotated_masks.append(rotated_mask)
                
                # Stack rotated masks
                target['masks'] = torch.stack(rotated_masks)
            
            # Rotate boxes if present
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                
                # Convert boxes to corners
                corners = []
                for box in boxes:
                    xmin, ymin, xmax, ymax = box
                    # Get all 4 corners of the box
                    corners.append(torch.tensor([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]], dtype=torch.float32))
                
                corners = torch.stack(corners)  # Shape: [N, 4, 2]
                
                # Move corners to origin (center of rotation)
                corners = corners - torch.tensor([center[0], center[1]], dtype=torch.float32)
                
                # Convert angle to radians
                angle_rad = angle * np.pi / 180
                
                # Rotation matrix - explicitly set dtype to float32
                rot_matrix = torch.tensor([
                    [np.cos(angle_rad), -np.sin(angle_rad)],
                    [np.sin(angle_rad), np.cos(angle_rad)]
                ], dtype=torch.float32)
                
                # Apply rotation to corners
                corners = torch.matmul(corners, rot_matrix.T)
                
                # Move corners back
                corners = corners + torch.tensor([center[0], center[1]], dtype=torch.float32)
                
                # Get new bounding boxes from rotated corners
                new_boxes = []
                for box_corners in corners:
                    xmin = torch.min(box_corners[:, 0])
                    ymin = torch.min(box_corners[:, 1])
                    xmax = torch.max(box_corners[:, 0])
                    ymax = torch.max(box_corners[:, 1])
                    new_boxes.append(torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32))
                
                target['boxes'] = torch.stack(new_boxes)
            
        return image, target

class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, prob=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = self.jitter(image)
        return image, target

class RandomScale:
    def __init__(self, scales, prob=0.5):
        self.scales = scales
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            scale = random.choice(self.scales)
            
            # Check if image is a tensor or PIL Image
            if isinstance(image, torch.Tensor):
                # For tensor, shape is [C, H, W]
                _, h, w = image.shape
                new_h, new_w = int(h * scale), int(w * scale)
            else:
                # For PIL Image
                w, h = image.width, image.height
                new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize image
            image = F.resize(image, (new_h, new_w))
            
            # Scale masks if present
            if 'masks' in target:
                masks = target['masks']
                scaled_masks = []
                
                for mask in masks:
                    # Convert to PIL Image
                    mask_pil = Image.fromarray(mask.cpu().numpy())
                    # Resize mask
                    scaled_mask_pil = F.resize(mask_pil, (new_h, new_w))
                    # Convert back to tensor
                    scaled_mask = torch.from_numpy(np.array(scaled_mask_pil))
                    scaled_masks.append(scaled_mask)
                
                # Stack scaled masks
                target['masks'] = torch.stack(scaled_masks)
            
            # Scale boxes if present
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                
                # Scale the boxes
                boxes[:, 0] = boxes[:, 0] * (new_w / w)  # xmin
                boxes[:, 1] = boxes[:, 1] * (new_h / h)  # ymin
                boxes[:, 2] = boxes[:, 2] * (new_w / w)  # xmax
                boxes[:, 3] = boxes[:, 3] * (new_h / h)  # ymax
                
                target['boxes'] = boxes
                
                # Update area if present
                if 'area' in target:
                    target['area'] = target['area'] * (scale * scale)
            
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

def get_transform(train):
    """
    Get transformation pipeline based on whether it's for training or evaluation.
    
    Args:
        train (bool): Whether the transforms are for training
        
    Returns:
        Compose: Composition of transforms
    """
    transforms = []
    
    if train:
        # Add training-specific augmentations (before ToTensor)
        transforms.append(RandomHorizontalFlip())
        transforms.append(RandomRotation(10))
        transforms.append(RandomScale([0.8, 0.9, 1.1, 1.2]))
        transforms.append(ColorJitter())
        logger.info("Created training transforms with augmentations: HorizontalFlip, Rotation(±10°), Scale(0.8-1.2), ColorJitter")
    else:
        logger.info("Created evaluation transforms (no augmentations)")
    
    # Add ToTensor transform after spatial transforms
    transforms.append(ToTensor())
    
    # Add normalization
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return Compose(transforms) 