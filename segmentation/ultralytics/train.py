from ultralytics import YOLO
from pathlib import Path
import torch
from loguru import logger
import os

def train_model(data_yaml, epochs=100, batch_size=16, imgsz=640, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train YOLOv8-seg model on PASCAL VOC dataset.
    
    Args:
        data_yaml (str): Path to data configuration file
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        imgsz (int): Image size
        device (str): Device to use for training
    """
    # Load a model
    model = YOLO('yolov8n-seg.pt')  # load a pretrained model
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        patience=50,  # early stopping patience
        save=True,  # save checkpoints
        save_period=10,  # save checkpoint every 10 epochs
        project='runs/train',  # save to project/name
        name='yolov8n-seg-voc',  # save to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        pretrained=True,  # use pretrained weights
        optimizer='auto',  # optimizer to use
        verbose=True,  # print results
        seed=0,  # random seed
        deterministic=True,  # deterministic training
        single_cls=False,  # train as multi-class dataset
        rect=False,  # rectangular training
        cos_lr=True,  # cosine LR scheduler
        close_mosaic=10,  # disable mosaic augmentation last 10 epochs
        resume=False,  # resume training
        amp=True,  # Automatic Mixed Precision (AMP) training
        fraction=1.0,  # dataset fraction to train on
        val=False,  # Disable validation during training to avoid empty tensor errors
        plots=True,  # save training plots
        cache=False,  # cache images for faster training
        workers=8,  # number of worker threads for data loading
        overlap_mask=True,  # allow overlapping masks for multi-class segmentation
        mask_ratio=4,  # mask downsampling ratio
        conf=0.25,  # confidence threshold
        iou=0.7,  # IoU threshold
        max_det=300,  # maximum detections per image
        half=False,  # use half precision
        dnn=False,  # use OpenCV DNN for ONNX inference
    )
    
    return results

if __name__ == '__main__':
    try:
        # Path to data configuration file
        data_yaml = Path(__file__).parent / 'data' / 'voc.yaml'
        
        # Verify data configuration file exists
        if not data_yaml.exists():
            raise FileNotFoundError(f"Data configuration file not found: {data_yaml}")
        
        # Train model
        results = train_model(
            data_yaml=str(data_yaml),
            epochs=10,
            batch_size=32,  # Reduced batch size for better stability
            imgsz=640
        )
        
        logger.info(f"Training completed. Results saved to {results.save_dir}")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise 