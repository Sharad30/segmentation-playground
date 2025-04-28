#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import argparse
import warnings
import os
import wandb
import yaml
from datetime import datetime
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def train_multiclass_segmentation(dataset_yaml, epochs=50, batch_size=16, imgsz=640, model_size='n', 
                                patience=5, workers=8, device='', max_plot_items=None, hide_plot_warnings=False,
                                use_wandb=True, wandb_project="segmentation-multiclass", wandb_name=None, 
                                wandb_entity=None, resume_wandb=False, lr=None, no_cos_lr=False, optimizer="auto",
                                log_lr_per_epoch=False):
    """
    Train a YOLOv8 model for multi-class instance segmentation.
    
    Args:
        dataset_yaml: Path to dataset YAML file
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Image size
        model_size: YOLOv8 model size (n, s, m, l, x)
        patience: Early stopping patience
        workers: Number of worker threads
        device: Device to train on ('cpu', '0', '0,1,2,3', etc.)
        max_plot_items: Maximum number of items to include in validation plots
        hide_plot_warnings: Whether to suppress plot limit warnings
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: W&B project name
        wandb_name: W&B run name (default: auto-generated based on model size and timestamp)
        wandb_entity: W&B entity (username or team name)
        resume_wandb: Whether to resume a previous W&B run
        lr: Fixed learning rate (None = use YOLOv8's default)
        no_cos_lr: Disable cosine learning rate scheduler
        log_lr_per_epoch: Whether to log the learning rate for each epoch
    """
    # Suppress warnings about plot limits if requested
    if hide_plot_warnings:
        warnings.filterwarnings("ignore", message=".*Limiting validation plots to first.*")
        os.environ['ULTRALYTICS_HIDE_PLOT_LIMITS'] = '1'
    
    # Validate model size
    if model_size not in ['n', 's', 'm', 'l', 'x']:
        raise ValueError(f"Invalid model size: {model_size}. Choose from 'n', 's', 'm', 'l', 'x'")
    
    # Load dataset configuration to get metadata
    dataset_info = {}
    try:
        with open(dataset_yaml, 'r') as f:
            dataset_info = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load dataset YAML for metadata: {e}")
    
    # Generate W&B run name if not provided
    if use_wandb and wandb_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include learning rate in run name if specified
        lr_suffix = ""
        if lr is not None:
            lr_suffix = f"_lr{lr}"
        wandb_name = f"multiclass-seg-{model_size}{lr_suffix}-{timestamp}"
    
    # Initialize W&B
    if use_wandb:
        # Enable wandb sync with YOLOv8
        os.environ["WANDB_PROJECT"] = wandb_project
        # Prevent dataset upload to W&B
        os.environ["WANDB_DISABLED_POLICY"] = "DatasetEvents"
        
        if wandb_entity:
            os.environ["WANDB_ENTITY"] = wandb_entity
        
        # For YOLOv8 to log metrics automatically
        os.environ["WANDB_LOG_MODEL"] = "true"
        os.environ["WANDB_WATCH"] = "all"
            
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            entity=wandb_entity,
            resume=resume_wandb,
            config={
                "model_size": model_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "imgsz": imgsz,
                "patience": patience,
                "dataset": dataset_yaml,
                "num_classes": dataset_info.get("nc", "unknown"),
                "class_names": dataset_info.get("names", "unknown"),
                "optimizer": optimizer,
                "cos_lr": not no_cos_lr,
                "learning_rate": lr
            }
        )
        print(f"W&B initialized: {wandb.run.name}")
    
    # Load a pre-trained YOLOv8 segmentation model
    model = YOLO(f'yolov8{model_size}-seg.pt')
    
    # Set training parameters
    train_args = {
        'data': dataset_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'patience': patience,  # early stopping patience
        'save': True,  # save checkpoints
        'save_period': 10,  # save checkpoint every 10 epochs
        'project': 'runs/train',
        'name': wandb_name if use_wandb else 'multiclass-seg',  
        'exist_ok': True,  # existing project/name ok, do not increment
        'pretrained': True,  # use pretrained weights
        'optimizer': optimizer,  # optimizer to use
        'verbose': True,  # print results
        'seed': 0,  # random seed
        'deterministic': True,  # deterministic training
        'single_cls': False,  # multi-class dataset
        'rect': False,  # rectangular training
        'cos_lr': not no_cos_lr,  # cosine LR scheduler (can be disabled)
        'close_mosaic': 10,  # disable mosaic augmentation last 10 epochs
        'resume': False,  # resume training
        'amp': True,  # Automatic Mixed Precision (AMP) training
        'fraction': 1.0,  # dataset fraction to train on
        'val': True,  # validate during training
        'plots': True,  # save training plots
        'cache': False,  # cache images for faster training
        'workers': workers,  # number of worker threads for data loading
        'overlap_mask': True,  # allow overlapping masks
        'mask_ratio': 4,  # mask downsampling ratio
        'device': device,  # device to train on
    }
    
    # Set learning rate if specified
    if lr is not None:
        train_args['lr0'] = lr  # initial learning rate
    
    # Add max_plot parameter if specified
    if max_plot_items is not None:
        train_args['max_plot'] = max_plot_items
    
    # Train the model
    results = model.train(**train_args)
    
    # If using W&B, explicitly log results from the CSV file YOLOv8 produces
    if use_wandb and results is not None:
        try:
            # Get the results CSV file path from the results object
            csv_path = results.save_dir / "results.csv"
            if csv_path.exists():
                # Read the CSV into a pandas DataFrame
                df = pd.read_csv(csv_path)
                
                # Extract learning rate information if log_lr_per_epoch is enabled
                if log_lr_per_epoch:
                    # Check if lr.npy exists
                    lr_file = results.save_dir / "lr.npy"
                    if not lr_file.exists():
                        # If lr.npy doesn't exist, try to find optimizer history in model's files
                        for file in results.save_dir.glob("*.pt"):
                            if "last" in file.name or "best" in file.name:
                                try:
                                    # Load the model to extract optimizer state
                                    checkpoint = torch.load(file, map_location='cpu')
                                    if 'optimizer' in checkpoint and 'param_groups' in checkpoint['optimizer']:
                                        # Extract learning rates from optimizer state
                                        lrs = [pg['lr'] for pg in checkpoint['optimizer']['param_groups']]
                                        # Use the first lr
                                        lr_values = [lrs[0] if lrs else None]
                                        np.save(str(lr_file), lr_values)
                                        break
                                except Exception as e:
                                    print(f"Warning: Could not extract learning rates from {file}: {e}")
                
                    # If lr.npy now exists, load and log learning rates
                    if lr_file.exists():
                        try:
                            lr_values = np.load(str(lr_file))
                            # Create a column for learning rates
                            df['learning_rate'] = None
                            # Fill in learning rate values
                            for i in range(min(len(df), len(lr_values))):
                                df.loc[i, 'learning_rate'] = lr_values[i]
                        except Exception as e:
                            print(f"Warning: Could not load learning rates from {lr_file}: {e}")
                    else:
                        # If we couldn't find learning rates, estimate from the initial learning rate
                        if lr is not None:
                            # For cosine scheduler, estimate learning rate per epoch
                            if not no_cos_lr:
                                # Cosine annealing formula: lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(t/T * pi))
                                # Where t is current epoch and T is total epochs
                                lr_min = lr * 0.01  # typical minimum LR is 1% of initial
                                df['learning_rate'] = [
                                    lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(epoch / epochs * np.pi))
                                    for epoch in df['epoch']
                                ]
                            else:
                                # Constant learning rate
                                df['learning_rate'] = lr
                        else:
                            # If no learning rate was specified, use default YOLOv8 values
                            default_lr = 0.01
                            if not no_cos_lr:
                                lr_min = default_lr * 0.01
                                df['learning_rate'] = [
                                    lr_min + 0.5 * (default_lr - lr_min) * (1 + np.cos(epoch / epochs * np.pi))
                                    for epoch in df['epoch']
                                ]
                            else:
                                df['learning_rate'] = default_lr
                
                # Log each epoch's metrics to wandb
                for index, row in df.iterrows():
                    epoch_metrics = {}
                    for col in df.columns:
                        # Skip non-metric columns like 'epoch' or empty values
                        if col != 'epoch' and not pd.isna(row[col]):
                            epoch_metrics[col] = row[col]
                    
                    # Add epoch number
                    epoch = int(row['epoch']) if 'epoch' in df.columns else index
                    
                    # Log to wandb
                    wandb.log(epoch_metrics, step=epoch)
                
                # Now create and log plots for key metrics
                if len(df) > 1:  # At least two points to plot
                    # Create box/mask mAP plot
                    if 'metrics/mAP50(B)' in df.columns and 'metrics/mAP50-95(B)' in df.columns:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='Box mAP50')
                        ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='Box mAP50-95')
                        if 'metrics/mAP50(M)' in df.columns:
                            ax.plot(df['epoch'], df['metrics/mAP50(M)'], label='Mask mAP50')
                        if 'metrics/mAP50-95(M)' in df.columns:
                            ax.plot(df['epoch'], df['metrics/mAP50-95(M)'], label='Mask mAP50-95')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('mAP')
                        ax.set_title('Object Detection & Segmentation Performance')
                        ax.legend()
                        ax.grid(True)
                        wandb.log({"plots/mAP_metrics": wandb.Image(fig)})
                        plt.close(fig)
                    
                    # Create loss plots
                    loss_cols = [col for col in df.columns if 'loss' in col.lower()]
                    if loss_cols:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        for col in loss_cols:
                            ax.plot(df['epoch'], df[col], label=col)
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title('Training Losses')
                        ax.legend()
                        ax.grid(True)
                        wandb.log({"plots/losses": wandb.Image(fig)})
                        plt.close(fig)
                    
                    # If learning rate is available, create a learning rate plot
                    if 'learning_rate' in df.columns:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(df['epoch'], df['learning_rate'])
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Learning Rate')
                        ax.set_title('Learning Rate Schedule')
                        ax.grid(True)
                        wandb.log({"plots/learning_rate": wandb.Image(fig)})
                        plt.close(fig)
                
                # Also upload the CSV as an artifact
                wandb.log_artifact(
                    str(csv_path), 
                    name=f"training_results_{wandb_name}", 
                    type="training-results",
                    description="YOLOv8 training results CSV"
                )
            
            # Log plots
            plots_dir = results.save_dir / "plots"
            if plots_dir.exists():
                for plot_file in plots_dir.glob("*.png"):
                    wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
        
        except Exception as e:
            print(f"Warning: Error while logging additional results to W&B: {e}")
    
    # Set validation parameters for final validation
    val_args = {
        'data': dataset_yaml,
    }
    
    # Add max_plot parameter if specified for validation too
    if max_plot_items is not None:
        val_args['max_plot'] = max_plot_items
    
    # Validate the model on the validation set
    print("\nRunning final validation...")
    metrics = model.val(**val_args)
    
    # Log final validation metrics to W&B
    if use_wandb:
        try:
            # Log key metrics to W&B
            box_map = metrics.box.map
            mask_map = metrics.seg.map
            wandb.log({
                "val/box_mAP": box_map,
                "val/mask_mAP": mask_map,
                "val/precision": metrics.seg.p,
                "val/recall": metrics.seg.r,
            })
            
            # Log confusion matrix if it exists
            conf_matrix_path = results.save_dir / "plots" / "confusion_matrix.png"
            if conf_matrix_path.exists():
                wandb.log({"plots/confusion_matrix": wandb.Image(str(conf_matrix_path))})
                
            # Log per-class metrics if available
            if hasattr(metrics.seg, 'ap_class_index'):
                for i, class_idx in enumerate(metrics.seg.ap_class_index):
                    class_name = metrics.names[class_idx]
                    class_map = metrics.seg.classes[i]
                    wandb.log({f"val/class_{class_name}_mAP": class_map})
                
            # Upload the best model as an artifact
            best_model_path = results.save_dir / "weights" / "best.pt"
            if best_model_path.exists():
                model_artifact = wandb.Artifact(
                    name=f"model_{wandb_name}", 
                    type="model",
                    description=f"Best YOLOv8 {model_size} model trained on {dataset_yaml}"
                )
                model_artifact.add_file(str(best_model_path))
                wandb.log_artifact(model_artifact)
            
            # Finish the W&B run
            wandb.finish()
        except Exception as e:
            print(f"Warning: Could not log final metrics to W&B: {e}")
            if use_wandb:
                wandb.finish()
    
    print(f"\nTraining completed. Results saved to {results.save_dir}")
    
    return results, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for multi-class instance segmentation")
    parser.add_argument("--dataset", type=str, default="datasets/yolo_voc_all_classes/dataset.yaml", 
                        help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model_size", type=str, default="n", choices=['n', 's', 'm', 'l', 'x'],
                        help="YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--device", type=str, default="", help="Device to train on ('cpu', '0', '0,1,2,3', etc.)")
    parser.add_argument("--max_plot_items", type=int, help="Maximum items per validation plot (default is YOLOv8's 50)")
    parser.add_argument("--hide_plot_warnings", action="store_true", help="Hide warnings about validation plot limits")
    # Learning rate parameters
    parser.add_argument("--lr", type=float, help="Initial learning rate (default: auto)")
    parser.add_argument("--no_cos_lr", action="store_true", help="Disable cosine learning rate scheduler")
    parser.add_argument("--log_lr", action="store_true", help="Log learning rate for each epoch")
    # W&B arguments
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="segmentation-multiclass", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, help="W&B run name (default: auto-generated)")
    parser.add_argument("--wandb_entity", type=str, help="W&B entity (username or team name)")
    parser.add_argument("--resume_wandb", action="store_true", help="Resume previous W&B run")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer to use (auto, AdamW, SGD, etc.)")
    
    args = parser.parse_args()
    
    # Train model
    train_multiclass_segmentation(
        dataset_yaml=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        model_size=args.model_size,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        max_plot_items=args.max_plot_items,
        hide_plot_warnings=args.hide_plot_warnings,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
        resume_wandb=args.resume_wandb,
        lr=args.lr,
        no_cos_lr=args.no_cos_lr,
        optimizer=args.optimizer,
        log_lr_per_epoch=args.log_lr
    ) 