# VOC Segmentation with YOLOv8

This project converts the PASCAL VOC dataset to YOLO format and trains YOLOv8 segmentation models. It supports both binary segmentation (person vs. background) and multi-class segmentation (all 20 VOC classes).

## Requirements

```bash
poetry install
```

## Download Pascal VOC data

```
curl -L "https://public.roboflow.com/ds/HTftkJJ2Hd?key=uSyWPGx3HN" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

## Dataset Conversion

### Convert yolo detection masks to yolo segmentation masks

Run the conversion command after having the images and labels in ultralytics yolo format like shown below:

```
voc_yolo_detection*/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

```
poetry run python segmentation/ultralytics/convert_detection_to_segmentation.py
```

## Training

### Multi-Class Segmentation Model

```bash
poetry run python segmentation/ultralytics/hyperparameter_tuner.py --data datasets/voc_yolo_segmentation/dataset.yaml --lr 0.001 0.0001 0.00001 --batch 16 32 64 --model_size x --device 0 --epochs 1 --wandb_project segmentation-hyp-tuning-lr-vs-bs
```

Training arguments:
- `--dataset`: Path to the dataset YAML file
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 16)
- `--imgsz`: Image size (default: 640)
- `--model_size`: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)

Training results and checkpoints are saved to `runs/train/.

## Inference

After training, you can use the model for inference:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/train/multiclass-seg/weights/best.pt')

# Perform inference on an image
results = model('path/to/image.jpg')

# Visualize results
results[0].show()

# Save results
results[0].save('results.jpg')
```



curl -L "https://universe.roboflow.com/ds/7rBmZBvwIm?key=D05ucjWbK4" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip