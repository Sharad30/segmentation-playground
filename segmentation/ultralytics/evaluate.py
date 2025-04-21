#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import torch
import argparse
from loguru import logger
import warnings
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import wandb
from datetime import datetime

def evaluate_model(model_path, dataset_yaml, batch_size=16, imgsz=640, 
                   device='cuda' if torch.cuda.is_available() else 'cpu',
                   conf=0.001, iou=0.6, max_det=300, 
                   plots=True, save_json=True, verbose=True,
                   max_plot_items=None, hide_plot_warnings=False,
                   use_wandb=False, wandb_project=None, wandb_name=None, 
                   wandb_entity=None, classes=None):
    """
    Evaluate a YOLOv8 segmentation model (supports single-class and multi-class).
    
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
        use_wandb: Whether to log results to wandb
        wandb_project: W&B project name
        wandb_name: W&B run name
        wandb_entity: W&B entity (username or team name)
        classes: List of specific class indices to evaluate (None = all classes)
        
    Returns:
        Evaluation metrics
    """
    # Suppress warnings about plot limits if requested
    if hide_plot_warnings:
        warnings.filterwarnings("ignore", message=".*Limiting validation plots to first.*")
        os.environ['ULTRALYTICS_HIDE_PLOT_LIMITS'] = '1'
    
    # Load dataset yaml to get model metadata
    dataset_info = {}
    try:
        with open(dataset_yaml, 'r') as f:
            dataset_info = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load dataset YAML for metadata: {e}")
    
    # Initialize wandb if requested
    if use_wandb:
        # Auto-generate run name if not provided
        if wandb_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = Path(model_path).stem
            wandb_name = f"eval-{model_name}-{timestamp}"
            
        # Set wandb project
        if wandb_project is None:
            wandb_project = "segmentation-evaluation"
            
        # Initialize wandb
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            entity=wandb_entity,
            config={
                "model_path": model_path,
                "dataset": dataset_yaml,
                "conf_threshold": conf,
                "iou_threshold": iou,
                "image_size": imgsz,
                "batch_size": batch_size,
                "max_detections": max_det,
                "num_classes": dataset_info.get("nc", "unknown"),
                "class_names": dataset_info.get("names", "unknown"),
                "specific_classes": classes if classes is not None else "all"
            }
        )
        logger.info(f"W&B initialized: {wandb.run.name}")
    
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
    
    # Add classes parameter if specified
    if classes is not None:
        val_args['classes'] = classes
    
    # Add max_plot parameter if specified
    if max_plot_items is not None:
        val_args['max_plot'] = max_plot_items
    
    # Evaluate the model
    metrics = model.val(**val_args)
    
    # Extract and visualize metrics
    results = {}
    
    # Extract key metrics
    try:
        results['box_map50'] = metrics.box.map50
        results['box_map'] = metrics.box.map
        results['mask_map50'] = metrics.seg.map50
        results['mask_map'] = metrics.seg.map
        results['precision'] = metrics.seg.p
        results['recall'] = metrics.seg.r
        
        # Get per-class metrics if available
        if hasattr(metrics.seg, 'ap_class_index'):
            class_names = []
            class_maps = []
            
            for i, class_idx in enumerate(metrics.seg.ap_class_index):
                class_name = metrics.names[class_idx]
                class_map = metrics.seg.classes[i]
                
                class_names.append(class_name)
                class_maps.append(class_map)
                
                results[f'mask_map_{class_name}'] = class_map
        
            # Create and save class-wise performance plot
            if class_names and len(class_names) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(class_names))
                ax.barh(y_pos, class_maps, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(class_names)
                ax.invert_yaxis()  # labels read top-to-bottom
                ax.set_xlabel('mAP50-95')
                ax.set_title('Class-wise Mask AP')
                ax.grid(axis='x', linestyle='--', alpha=0.6)
                
                # Save plot
                class_plot_path = Path(model_path).parent / "class_performance.png"
                plt.tight_layout()
                plt.savefig(class_plot_path)
                plt.close(fig)
                
                # Log to wandb if enabled
                if use_wandb:
                    wandb.log({"plots/class_map": wandb.Image(str(class_plot_path))})
    
    except Exception as e:
        logger.warning(f"Error extracting metrics: {e}")
    
    # Print metrics
    if verbose:
        try:
            logger.info(f"\nModel: {model_path}")
            logger.info(f"Dataset: {dataset_yaml}")
            logger.info(f"Box mAP50: {results['box_map50']:.4f}")
            logger.info(f"Box mAP50-95: {results['box_map']:.4f}")
            logger.info(f"Mask mAP50: {results['mask_map50']:.4f}")
            logger.info(f"Mask mAP50-95: {results['mask_map']:.4f}")
            logger.info(f"Precision: {results['precision']:.4f}")
            logger.info(f"Recall: {results['recall']:.4f}")
            
            # Print per-class metrics if available
            if hasattr(metrics.seg, 'ap_class_index'):
                logger.info("\nPer-class mask mAP50-95:")
                for i, class_idx in enumerate(metrics.seg.ap_class_index):
                    class_name = metrics.names[class_idx]
                    class_map_value = metrics.seg.classes[i]
                    logger.info(f"  {class_name}: {class_map_value:.4f}")
                
        except Exception as e:
            logger.warning(f"Detailed metrics printing failed: {e}")
    
    # Log results to wandb
    if use_wandb:
        try:
            # Log metrics
            wandb.log(results)
            
            # Log confusion matrix if it exists
            confusion_matrix_path = Path(model_path).parent / "confusion_matrix.png"
            if confusion_matrix_path.exists():
                wandb.log({"plots/confusion_matrix": wandb.Image(str(confusion_matrix_path))})
            
            # Log PR curve if it exists
            pr_curve_path = Path(model_path).parent / "PR_curve.png"
            if pr_curve_path.exists():
                wandb.log({"plots/pr_curve": wandb.Image(str(pr_curve_path))})
            
            # Log F1 curve if it exists
            f1_curve_path = Path(model_path).parent / "F1_curve.png"
            if f1_curve_path.exists():
                wandb.log({"plots/f1_curve": wandb.Image(str(f1_curve_path))})
            
            # Log P curve if it exists
            p_curve_path = Path(model_path).parent / "P_curve.png"
            if p_curve_path.exists():
                wandb.log({"plots/p_curve": wandb.Image(str(p_curve_path))})
            
            # Log R curve if it exists
            r_curve_path = Path(model_path).parent / "R_curve.png"
            if r_curve_path.exists():
                wandb.log({"plots/r_curve": wandb.Image(str(r_curve_path))})
            
            # Finish wandb run
            wandb.finish()
            logger.info("Results logged to wandb")
            
        except Exception as e:
            logger.warning(f"Error logging to wandb: {e}")
            wandb.finish()
    
    return metrics, results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 segmentation model")
    parser.add_argument("--model", type=str, default="runs/train/instance-seg/weights/best.pt",
                      help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="datasets/yolo_voc_instance/dataset.yaml",
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
    parser.add_argument("--classes", type=str, help="Comma-separated list of specific class indices to evaluate")
    # W&B arguments
    parser.add_argument("--wandb", action="store_true", help="Log results to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="segmentation-evaluation", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, help="W&B run name (default: auto-generated)")
    parser.add_argument("--wandb_entity", type=str, help="W&B entity (username or team name)")
    
    args = parser.parse_args()
    
    # Parse classes if provided
    classes = None
    if args.classes:
        try:
            classes = [int(c.strip()) for c in args.classes.split(',')]
        except Exception as e:
            logger.warning(f"Error parsing classes argument: {e}")
    
    # Evaluate model
    metrics, results = evaluate_model(
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
        hide_plot_warnings=args.hide_plot_warnings,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
        classes=classes
    )
    
    logger.info(f"Evaluation completed.") 