# PyTorch Semantic Segmentation Framework

This is a framework for training UNet models for semantic segmentation using PyTorch.

## Overview

The framework includes:

- UNet model implementation
- COCO format dataset loader
- Training and validation loops
- Metrics calculation (pixel accuracy, Dice coefficient)
- Wandb integration for experiment tracking
- Configurable data augmentation
- Logging with loguru

## Directory Structure

```
segmentation/pytorch/
├── __init__.py           # Package initialization
├── augmentations.py      # Data augmentation functions
├── config.yaml           # Default configuration
├── dataset.py            # Dataset and data loader implementation
├── model.py              # UNet model implementation
├── run_training.py       # Script to start training
├── train.py              # Training and validation functions
├── output/               # Model checkpoints (created during training)
└── logs/                 # Training logs (created during training)
```

## Requirements

- PyTorch
- torchvision
- pycocotools
- wandb
- loguru
- PyYAML
- tqdm
- numpy
- Pillow

## Training

### Using Default Configuration

Run training with the default configuration:

```bash
cd segmentation/pytorch
python run_training.py
```

This will load the default configuration from `config.yaml`.

### Custom Configuration

To use a custom configuration, create a new YAML file and pass it:

```bash
python run_training.py --config my_config.yaml
```

## Configuration Options

The `config.yaml` file includes the following options:

### Dataset Configuration
- `data_dir`: Path to the dataset directory
- `num_classes`: Number of segmentation classes (including background)
- `input_size`: Size to resize input images to
- `batch_size`: Batch size for training
- `num_workers`: Number of worker processes for data loading

### Model Configuration
- `model_type`: Model architecture ("unet")
- `n_channels`: Number of input channels (3 for RGB)
- `bilinear`: Whether to use bilinear upsampling

### Training Configuration
- `epochs`: Number of training epochs
- `learning_rate`: Fixed learning rate with AdamW optimizer
- `weight_decay`: L2 regularization strength
- `use_lr_scheduler`: Whether to use learning rate scheduler
- `min_lr`: Minimum learning rate for scheduler

### Augmentation Configuration
- `brightness`, `contrast`, `saturation`, `hue`: Color jitter parameters

### Technical Configuration
- `seed`: Random seed for reproducibility
- `use_amp`: Whether to use mixed precision training

### Output Configuration
- `output_dir`: Directory to save model checkpoints
- `log_dir`: Directory to save logs
- `run_name`: Name for the current run

### Wandb Configuration
- `use_wandb`: Whether to use Weights & Biases
- `wandb_project`: Wandb project name

## Dataset Format

The framework expects the dataset to be in COCO format with the following structure:

```
dataset_root/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

Each annotation file should contain segmentation polygons in COCO format.

## Model Checkpoints

During training, two model checkpoints are saved:
- `checkpoint_latest.pth`: The most recent model
- `checkpoint_best.pth`: The model with the best validation Dice score 