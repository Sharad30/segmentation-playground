# VOC Segmentation with YOLOv8

This project converts the PASCAL VOC dataset to YOLO format and trains YOLOv8 segmentation models. It supports both binary segmentation (person vs. background) and multi-class segmentation (all 20 VOC classes).

## Requirements

```bash
pip install ultralytics opencv-python numpy tqdm matplotlib
```

## Dataset Conversion

### Binary Segmentation (Person Only)

The `convert_voc_to_yolo_fixed.py` script converts the PASCAL VOC dataset to YOLO format with binary person segmentation masks:

```bash
python convert_voc_to_yolo_fixed.py --voc_path data/VOCdevkit/VOC2012 --output_path data/yolo_voc_person
```

### Multi-Class Segmentation (All Classes)

The `convert_voc_to_yolo_multiclass.py` script preserves all class information from the PASCAL VOC dataset:

```bash
python convert_voc_to_yolo_multiclass.py --voc_path data/VOCdevkit/VOC2012 --output_path data/yolo_voc_multiclass
```

Both scripts create a dataset with the following structure:

```
data/yolo_voc_*/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

## Visualization

### Binary Mask Visualization

To visualize binary person segmentation masks:

```bash
python visualize_mask.py --dataset data/yolo_voc_person --image 2007_006477
```

### Multi-Class Mask Visualization

To visualize multi-class segmentation masks with different colors for each class:

```bash
python visualize_mask_multiclass.py --dataset data/yolo_voc_multiclass --image 2007_006477
```

Both scripts create visualizations in the `visualizations` directory, showing:
- The original image
- The segmentation mask
- The overlay of the mask on the image

## Training

### Binary Segmentation Model

```bash
python train_person_segmentation.py --dataset data/yolo_voc_person/dataset.yaml --epochs 50 --batch_size 16 --imgsz 640 --model_size n
```

### Multi-Class Segmentation Model

```bash
python train_multiclass_segmentation.py --dataset data/yolo_voc_multiclass/dataset.yaml --epochs 50 --batch_size 16 --imgsz 640 --model_size n
```

Training arguments:
- `--dataset`: Path to the dataset YAML file
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 16)
- `--imgsz`: Image size (default: 640)
- `--model_size`: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)

Training results and checkpoints are saved to `runs/train/person-seg/` or `runs/train/multiclass-seg/`.

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

## Notes

- For binary segmentation, masks have pixel values of 0 (background) and 1 (person).
- For multi-class segmentation, masks have pixel values from 0 to 20, corresponding to the 21 VOC classes (including background).
- Binary segmentation is faster to train but only recognizes one class.
- Multi-class segmentation is more challenging but can distinguish between all object types.
- YOLOv8 expects segmentation masks to be saved in PNG format with the same name as the corresponding image.