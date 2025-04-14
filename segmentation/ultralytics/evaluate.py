from ultralytics import YOLO
from pathlib import Path
import torch
from loguru import logger

def evaluate_model(model_path, data_yaml, batch_size=16, imgsz=640, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate YOLOv8-seg model on PASCAL VOC dataset.
    
    Args:
        model_path (str): Path to trained model
        data_yaml (str): Path to data configuration file
        batch_size (int): Batch size
        imgsz (int): Image size
        device (str): Device to use for evaluation
    """
    # Load a model
    model = YOLO(model_path)
    
    # Evaluate the model
    metrics = model.val(
        data=data_yaml,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        split='val',  # validation split
        save_json=True,  # save results to JSON
        save_hybrid=True,  # save hybrid version of labels
        conf=0.001,  # confidence threshold
        iou=0.6,  # IoU threshold
        max_det=300,  # maximum detections per image
        half=True,  # use half precision
        plots=True,  # save plots
    )
    
    return metrics

if __name__ == '__main__':
    # Path to trained model
    model_path = 'runs/train/yolov8n-seg-voc/weights/best.pt'
    
    # Path to data configuration file
    data_yaml = Path(__file__).parent / 'data' / 'voc.yaml'
    
    # Evaluate model
    metrics = evaluate_model(
        model_path=model_path,
        data_yaml=str(data_yaml),
        batch_size=16,
        imgsz=640
    )
    
    logger.info(f"Evaluation completed. Metrics: {metrics}") 