from ultralytics import YOLO
from pathlib import Path
import torch
from loguru import logger
import cv2
import numpy as np

def predict_image(model_path, image_path, conf_thres=0.25, iou_thres=0.45, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Make predictions on a single image using YOLOv8-seg model.
    
    Args:
        model_path (str): Path to trained model
        image_path (str): Path to input image
        conf_thres (float): Confidence threshold
        iou_thres (float): IoU threshold
        device (str): Device to use for prediction
    """
    # Load a model
    model = YOLO(model_path)
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return None
    
    # Make prediction
    results = model(
        image,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        max_det=300,
        agnostic_nms=False,
        classes=None,  # filter by class
        retina_masks=True,  # use high-resolution segmentation masks
    )
    
    return results

def visualize_prediction(image_path, results, save_path=None):
    """
    Visualize prediction results.
    
    Args:
        image_path (str): Path to input image
        results: Prediction results from model
        save_path (str, optional): Path to save visualization
    """
    # Read image
    image = cv2.imread(str(image_path))
    
    # Plot results
    plotted = results[0].plot()
    
    # Save or show
    if save_path:
        cv2.imwrite(str(save_path), plotted)
        logger.info(f"Saved visualization to {save_path}")
    else:
        cv2.imshow('Prediction', plotted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Path to trained model
    model_path = 'runs/train/yolov8n-seg-voc/weights/best.pt'
    
    # Path to test image
    image_path = '/home/ubuntu/sharad/segmentation-playground/data/VOCdevkit/VOC2012/JPEGImages/2007_000392.jpg'
    
    # Make prediction
    results = predict_image(
        model_path=model_path,
        image_path=image_path,
        conf_thres=0.25,
        iou_thres=0.45
    )
    
    if results:
        # Visualize prediction
        save_path = 'runs/predict/prediction.jpg'
        visualize_prediction(image_path, results, save_path) 