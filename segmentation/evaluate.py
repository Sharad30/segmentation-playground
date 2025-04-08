import os
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile

from data_splits import create_data_splits
from model import get_instance_segmentation_model

class Evaluator:
    def __init__(self, model_path, config=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to the trained model checkpoint
            config (dict, optional): Configuration dictionary. If None, load from checkpoint.
        """
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get config
        if config is None:
            self.config = checkpoint['config']
        else:
            self.config = config
        
        # Set device
        self.device = torch.device(self.config['device'])
        
        # Create model
        self.model = get_instance_segmentation_model(
            num_classes=self.config['num_classes'],
            pretrained=False
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Create data loaders
        self.dataloaders = create_data_splits(
            self.config['data_root'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers']
        )
        
        logger.info(f"Evaluator initialized with model from {model_path}")
    
    def evaluate(self, dataset_type='test'):
        """
        Evaluate the model on the specified dataset.
        
        Args:
            dataset_type (str): Dataset to evaluate on ('train', 'val', or 'test')
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_type} dataset")
        
        dataloader = self.dataloaders[dataset_type]
        coco_results = []
        
        for images, targets in tqdm(dataloader, desc=f"Evaluating {dataset_type}"):
            images = [img.to(self.device) for img in images]
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(images)
            
            # Process outputs
            for i, (output, target) in enumerate(zip(outputs, targets)):
                image_id = target['image_id'].item()
                
                if 'boxes' in output:
                    boxes = output['boxes']
                    scores = output['scores']
                    labels = output['labels']
                    masks = output['masks']
                    
                    # Convert to COCO format
                    for box, score, label, mask in zip(boxes, scores, labels, masks):
                        if score < 0.1:  # Lower threshold from 0.5 to 0.1
                            continue
                            
                        # Convert box to [x, y, width, height] format
                        x, y, x2, y2 = box.tolist()
                        width = x2 - x
                        height = y2 - y
                        
                        # Convert mask to RLE format
                        mask = mask > 0.5
                        mask = mask.squeeze().cpu().numpy().astype(np.uint8)
                        
                        # Create COCO annotation
                        result = {
                            'image_id': image_id,
                            'category_id': label.item(),
                            'bbox': [x, y, width, height],
                            'score': score.item(),
                            'segmentation': self._mask_to_rle(mask)
                        }
                        
                        coco_results.append(result)
        
        # Save results
        results_file = os.path.join(self.config['output_dir'], f'{dataset_type}_results.json')
        with open(results_file, 'w') as f:
            json.dump(coco_results, f)
        
        logger.info(f"Evaluation results saved to {results_file}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(coco_results, dataset_type)
        
        return metrics
    
    def _mask_to_rle(self, mask):
        """
        Convert a binary mask to RLE format.
        
        Args:
            mask (np.ndarray): Binary mask
            
        Returns:
            dict: RLE encoded mask
        """
        # Simple RLE encoding
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return {'counts': runs.tolist(), 'size': list(mask.shape)}
    
    def _calculate_metrics(self, results, dataset_type):
        """
        Calculate evaluation metrics using COCO API.
        
        Args:
            results (list): List of detection results in COCO format
            dataset_type (str): Dataset type
            
        Returns:
            dict: Evaluation metrics
        """
        # For simplicity, we'll just return basic metrics
        # In a real implementation, you would use the COCO API to calculate mAP, etc.
        metrics = {
            'num_detections': len(results),
            'dataset_type': dataset_type
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

def main():
    """Main function to run the evaluation."""
    # Define configuration
    config = {
        'data_root': './data',
        'output_dir': './output_person',  # Updated output directory
        'num_classes': 2,  # Background + person
        'batch_size': 1,
        'num_workers': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # If there's a VOC_CLASSES list, update it
    VOC_CLASSES = [
        'background', 'person'
    ]
    
    # Initialize evaluator
    model_path = './output_person/best_model.pth'
    evaluator = Evaluator(model_path, config)
    
    # Evaluate model
    metrics = evaluator.evaluate(dataset_type='test')
    
    # Save metrics
    metrics_file = os.path.join(config['output_dir'], 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f)
    
    logger.info(f"Evaluation metrics saved to {metrics_file}")

if __name__ == "__main__":
    main() 