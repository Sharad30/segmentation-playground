import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from PIL import Image
import numpy as np
from loguru import logger
import random
from scipy import ndimage

class VOCInstanceSegmentationDataset(Dataset):
    def __init__(self, root, year='2012', image_set='train', transform=None):
        """
        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, 'train', 'val' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.voc = VOCSegmentation(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform
        logger.info(f"Loaded VOC{year} {image_set} dataset with {len(self.voc)} images")
        
    def __len__(self):
        return len(self.voc)
    
    def __getitem__(self, idx):
        img, mask = self.voc[idx]
        
        # Convert mask to numpy array for processing
        mask_np = np.array(mask)
        
        # For person detection, we only care about class 15 (person in VOC)
        # Create a binary mask where 1 = person, 0 = everything else
        person_mask = (mask_np == 15).astype(np.uint8)
        
        # Find connected components to separate individual people
        labeled_mask, num_features = ndimage.label(person_mask)
        
        # Create instance masks
        instance_masks = []
        for i in range(1, num_features + 1):  # Skip 0 (background)
            instance_mask = (labeled_mask == i).astype(np.uint8)
            instance_masks.append(instance_mask)
        
        # If no instances found, create a dummy mask
        if len(instance_masks) == 0:
            instance_masks = [np.zeros_like(mask_np, dtype=np.uint8)]
            
        # Stack instance masks along a new dimension
        instance_masks = np.stack(instance_masks, axis=0)
        
        # Create target dictionary - all instances are class 1 (person)
        target = {
            'masks': torch.as_tensor(instance_masks, dtype=torch.uint8),
            'image_id': torch.tensor([idx]),
            'labels': torch.ones((len(instance_masks),), dtype=torch.int64),  # All are class 1 (person)
            'area': torch.tensor([m.sum() for m in instance_masks], dtype=torch.float32),
            'iscrowd': torch.zeros((len(instance_masks),), dtype=torch.int64)
        }
        
        # Calculate bounding boxes
        boxes = []
        for mask in instance_masks:
            pos = np.where(mask)
            if len(pos[0]) > 0:  # Check if mask is not empty
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                
                # Ensure boxes have positive width and height
                # Add a small padding if width or height is zero
                if xmax == xmin:
                    xmax = xmin + 1
                if ymax == ymin:
                    ymax = ymin + 1
                    
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                # For empty masks, create a small valid box
                boxes.append([0, 0, 1, 1])  # Dummy box for empty mask
        
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        
        if self.transform:
            img, target = self.transform(img, target)
            
        # Final validation: ensure all boxes have positive width and height
        # This handles cases where the transform might have affected the boxes
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
        
        return img, target

    # Add a method to map VOC pixel values to class IDs
    def _get_class_id(self, pixel_value):
        # Map VOC pixel values to class IDs (1-20)
        # This is a simplified mapping - you'll need to use the actual VOC mapping
        return (pixel_value % 20) + 1

def get_voc_dataloader(root, year='2012', image_set='train', batch_size=2, transform=None, 
                       num_workers=4, shuffle=True, max_samples=None):
    """
    Create a DataLoader for VOC instance segmentation.
    
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, 'train', 'val' or 'test'
        batch_size (int): Batch size for the dataloader
        transform (callable, optional): Optional transform to be applied on a sample.
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the dataset
        max_samples (int, optional): Maximum number of samples to use. If None, use all samples.
    """
    dataset = VOCInstanceSegmentationDataset(root=root, year=year, image_set=image_set, transform=transform)
    
    # Limit the number of samples if specified
    if max_samples is not None and max_samples < len(dataset):
        # Create a subset of the dataset
        indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(indices)
        indices = indices[:max_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
        logger.info(f"Limited {image_set} dataset to {max_samples} samples")
    
    # Collate function to handle variable number of instances
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    logger.info(f"Created dataloader for {image_set} with batch size {batch_size}")
    return dataloader 