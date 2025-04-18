#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import torch
import argparse
from loguru import logger
import warnings
import os

def evaluate_multiclass_segmentation(model_path, dataset_yaml, batch_size=16, imgsz=640, 
                                   device='cuda' if torch.cuda.is_available() else 'cpu',
                                   conf=0.001, iou=0.6, max_det=300, 
                                   plots=True, save_json=True, verbose=True,
                                   max_plot_items=None, hide_plot_warnings=False):
    """
    Evaluate a YOLOv8 multi-class instance segmentation model.
    
    Args:
        model_path: Path to trained model
        dataset_yaml: Path to dataset YAML file
        batch_size: Batch size
        imgsz: Image size
        device: Device to use for evaluation ('cpu', '0', '0,1,2,3', etc.)
        conf: Confidence threshold
        iou: IoU threshold
        max_det: Maximum detections per image
        plots: Generate plots
        save_json: Save results to JSON
        verbose: Print detailed information
        max_plot_items: Maximum number of items to include in validation plots
        hide_plot_warnings: Whether to suppress plot limit warnings
        
    Returns:
        Evaluation metrics
    """
    # Suppress warnings about plot limits if requested
    if hide_plot_warnings:
        warnings.filterwarnings("ignore", message=".*Limiting validation plots to first.*")
        os.environ['ULTRALYTICS_HIDE_PLOT_LIMITS'] = '1'
    
    # Load the model
    model = YOLO(model_path)
    
    # Set validation parameters
    val_args = {
        'data': dataset_yaml,
        'batch': batch_size,
        'imgsz': imgsz,
        'device': device,
        'split': 'val',
        'save_json': save_json,
        'save_hybrid': True,
        'conf': conf,
        'iou': iou,
        'max_det': max_det,
        'half': True,
        'plots': plots,
        'verbose': verbose
    }
    
    # Add max_plot parameter if specified
    if max_plot_items is not None:
        val_args['max_plot'] = max_plot_items
    
    # Evaluate the model
    metrics = model.val(**val_args)
    
    # Extract and print key instance segmentation metrics
    if verbose:
        try:
            # Extract key metrics
            box_map = metrics.box.map  # Box mAP
            mask_map = metrics.seg.map  # Mask mAP
            
            logger.info(f"Model: {model_path}")
            logger.info(f"Dataset: {dataset_yaml}")
            logger.info(f"Box mAP50-95: {box_map:.4f}")
            logger.info(f"Mask mAP50-95: {mask_map:.4f}")
            
            # Print per-class metrics if available
            if hasattr(metrics.seg, 'ap_class_index'):
                logger.info("\nPer-class mask mAP50-95:")
                for i, class_idx in enumerate(metrics.seg.ap_class_index):
                    class_name = metrics.names[class_idx]
                    class_map_value = metrics.seg.classes[i]
                    logger.info(f"  {class_name}: {class_map_value:.4f}")
                
            # Print precision and recall if available
            if hasattr(metrics.seg, 'p') and hasattr(metrics.seg, 'r'):
                logger.info(f"\nMask Precision: {metrics.seg.p:.4f}")
                logger.info(f"Mask Recall: {metrics.seg.r:.4f}")
                
        except Exception as e:
            if verbose:
                logger.warning(f"Detailed metrics extraction failed: {e}")
            logger.info("See raw metrics below:")
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 multi-class instance segmentation model")
    parser.add_argument("--model", type=str, default="runs/train/multiclass-seg/weights/best.pt",
                      help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="datasets/yolo_voc_all_classes/dataset.yaml",
                      help="Path to dataset YAML file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default='', help="Device to use ('', 'cpu', '0', '0,1')")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    parser.add_argument("--max_det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument("--no_plots", action="store_false", dest="plots", help="Don't generate plots")
    parser.add_argument("--no_save_json", action="store_false", dest="save_json", help="Don't save results to JSON")
    parser.add_argument("--quiet", action="store_false", dest="verbose", help="Run without verbose output")
    parser.add_argument("--max_plot_items", type=int, help="Maximum items per validation plot (default is YOLOv8's 50)")
    parser.add_argument("--hide_plot_warnings", action="store_true", help="Hide warnings about validation plot limits")
    
    args = parser.parse_args()
    
    # Evaluate model
    metrics = evaluate_multiclass_segmentation(
        model_path=args.model,
        dataset_yaml=args.dataset,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        plots=args.plots,
        save_json=args.save_json,
        verbose=args.verbose,
        max_plot_items=args.max_plot_items,
        hide_plot_warnings=args.hide_plot_warnings
    )
    
    logger.info(f"Evaluation completed. Raw metrics: {metrics}") 