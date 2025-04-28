# Dataset Setup for YOLOv8 Training

This README explains how to set up custom datasets for YOLOv8 segmentation training using the provided script.

## Quick Start

1. Run the dataset setup script:
   ```bash
   ./setup_dataset.sh
   ```

2. Follow the prompts:
   - Enter a Roboflow dataset URL (or press Enter for the default Buildings dataset)
   - Choose a name for your dataset folder
   - Decide whether to keep or remove the original extracted files

3. The script will:
   - Download the dataset from Roboflow
   - Organize it into the correct YOLO format
   - Create a proper `dataset.yaml` file
   - Place everything in `datasets/YOUR_DATASET_NAME/`

## Using Your Custom Dataset

After setup, you can use your dataset for:

### Training
```bash
python segmentation/ultralytics/train.py --dataset datasets/YOUR_DATASET_NAME/dataset.yaml --epochs 20 --batch_size 16 --model_size n
```

### Visualization
```bash
python segmentation/ultralytics/visualize_yolo_dataset.py --data datasets/YOUR_DATASET_NAME/dataset.yaml
```

## Manual Dataset Setup

If you prefer to set up your dataset manually, here's the expected structure:

```
datasets/YOUR_DATASET_NAME/
├── dataset.yaml       # Configuration file
├── images/
│   ├── train/         # Training images
│   ├── val/           # Validation images
│   └── test/          # Test images
└── labels/
    ├── train/         # Training labels (YOLO format)
    ├── val/           # Validation labels (YOLO format)
    └── test/          # Test labels (YOLO format)
```

Your `dataset.yaml` should contain:

```yaml
path: /absolute/path/to/datasets/YOUR_DATASET_NAME
train: images/train
val: images/val
test: images/test

nc: 1  # Number of classes
names: ['ObjectName']  # Class names
```

## Getting Datasets from Roboflow

1. Go to [Roboflow Universe](https://universe.roboflow.com/)
2. Find a dataset with segmentation masks
3. Click "Download" and select:
   - Format: YOLO v5 PyTorch
   - Split: Yes (train/valid/test)
   - Download API Code: Copy the curl command

Use the downloaded URL with the setup script, or extract and organize manually.

## Troubleshooting

- **Error**: "Expected directories not found after extraction"
  - Make sure the Roboflow dataset is in YOLO format
  - Check that the downloaded zip has train/valid/test directories

- **Missing labels**:
  - Confirm your dataset has segmentation masks, not just bounding boxes

- **Training errors**:
  - Verify dataset.yaml has the correct paths and class information
  - Check that labels are in the proper YOLO segmentation format 