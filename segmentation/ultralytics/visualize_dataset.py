import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
import yaml
import os
from loguru import logger
import numpy as np
from PIL import Image
from collections import defaultdict

def load_dataset_config(config_path):
    """Load dataset configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_mask_classes(mask_path):
    """Get unique classes present in a mask."""
    mask = np.array(Image.open(mask_path))
    return np.unique(mask).tolist()

def create_fiftyone_dataset(config, dataset_name="voc-segmentation"):
    """Create a FiftyOne dataset from the VOC dataset."""
    # Delete existing dataset if it exists
    if dataset_name in fo.list_datasets():
        logger.info(f"Deleting existing dataset '{dataset_name}'...")
        fo.delete_dataset(dataset_name)
    
    # Create new dataset
    dataset = fo.Dataset(dataset_name)
    
    # Get paths from config
    base_path = config['path']
    train_dir = os.path.join(base_path, config['train'])
    mask_dir = os.path.join(base_path, config['mask_dir'])
    
    # Read split files
    train_split_path = os.path.join(base_path, config['train_split'])
    val_split_path = os.path.join(base_path, config['val_split'])
    
    with open(train_split_path, 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    
    with open(val_split_path, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    
    # Convert numeric keys to string keys in names dictionary
    mask_targets = {str(k): v for k, v in config['names'].items()}
    
    # Add samples to dataset
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for img_id in ids:
            # Construct paths
            img_path = os.path.join(train_dir, f"{img_id}{config['img_ext']}")
            mask_path = os.path.join(mask_dir, f"{img_id}{config['mask_ext']}")
            
            # Create sample
            sample = fo.Sample(filepath=img_path)
            
            # Add segmentation mask
            if os.path.exists(mask_path):
                sample["ground_truth"] = fo.Segmentation(
                    mask_path=mask_path,
                    mask_targets=mask_targets
                )
                
                # Add class tags based on mask content
                classes = get_mask_classes(mask_path)
                for class_id in classes:
                    class_name = mask_targets.get(str(class_id))
                    if class_name:
                        sample.tags.append(class_name)
            
            # Add split tag
            sample.tags.append(split)
            
            # Add sample to dataset
            dataset.add_sample(sample)
    
    return dataset

def print_class_statistics(dataset):
    """Print detailed statistics about class distribution."""
    # Get all class names from the dataset
    class_names = set()
    for sample in dataset:
        if "ground_truth" in sample and sample.ground_truth:
            class_names.update(sample.ground_truth.mask_targets.values())
    
    # Initialize counters
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    
    # Count occurrences in train and val sets
    for sample in dataset:
        if "ground_truth" in sample and sample.ground_truth:
            for class_name in sample.ground_truth.mask_targets.values():
                if "train" in sample.tags:
                    train_counts[class_name] += 1
                if "val" in sample.tags:
                    val_counts[class_name] += 1
    
    # Print statistics
    logger.info("\nClass Distribution Statistics:")
    logger.info(f"{'Class':<15} {'Train Count':<12} {'Val Count':<12} {'Total':<12}")
    logger.info("-" * 50)
    
    for class_name in sorted(class_names):
        train_count = train_counts[class_name]
        val_count = val_counts[class_name]
        total = train_count + val_count
        logger.info(f"{class_name:<15} {train_count:<12} {val_count:<12} {total:<12}")

def visualize_dataset(dataset):
    """Visualize the dataset in FiftyOne."""
    # Create a view with all samples
    view = dataset.view()
    
    # Print dataset statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Train samples: {len(dataset.match_tags('train'))}")
    logger.info(f"Validation samples: {len(dataset.match_tags('val'))}")
    
    # Print detailed class statistics
    print_class_statistics(dataset)
    
    # Launch the FiftyOne app with the view
    session = fo.launch_app(view)
    
    # Wait for the app to close
    session.wait()

def main():
    try:
        # Load configuration
        config_path = Path(__file__).parent / 'data' / 'voc.yaml'
        config = load_dataset_config(config_path)
        
        # Create and visualize dataset
        logger.info("Creating FiftyOne dataset...")
        dataset = create_fiftyone_dataset(config)
        
        # Visualize dataset
        logger.info("Launching FiftyOne visualization...")
        visualize_dataset(dataset)
        
    except Exception as e:
        logger.error(f"Error during dataset visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main() 