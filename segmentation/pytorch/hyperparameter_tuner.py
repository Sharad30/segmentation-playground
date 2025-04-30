#!/usr/bin/env python3
import argparse
import itertools
import os
import time
from pathlib import Path
import wandb
import yaml
import datetime
import sys
import traceback as tb
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from segmentation.pytorch.train import train

def run_hyperparameter_tuning(
    config_path,
    learning_rates=[1e-3, 5e-4, 1e-4],
    batch_sizes=[8, 16, 32],
    epochs=20,
    wandb_project="unet-hyperparameter-tuning",
    wandb_entity=None
):
    """
    Run multiple training experiments with different hyperparameters.
    
    Args:
        config_path: Path to base config YAML file
        learning_rates: List of learning rates to try
        batch_sizes: List of batch sizes to try
        epochs: Number of training epochs for each run
        wandb_project: W&B project name
        wandb_entity: W&B entity (username or team name)
    """
    # Create experiment directory with unique name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"hyperparameter_tuning_{timestamp}"
    
    # Create experiment root directory
    experiment_dir = Path("artifacts/pytorch") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created experiment directory: {experiment_dir}")
    
    # Create summary subdirectory for plots and metrics
    summary_dir = experiment_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    # Load base config
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Store results for creating a summary at the end
    all_results = []
    
    # Generate all combinations of hyperparameters
    all_combinations = list(itertools.product(learning_rates, batch_sizes))
    total_runs = len(all_combinations)
    
    print(f"Starting hyperparameter tuning with {total_runs} combinations")
    
    # Save hyperparameter config to experiment directory
    config = {
        "timestamp": timestamp,
        "base_config": config_path,
        "learning_rates": learning_rates,
        "batch_sizes": batch_sizes,
        "epochs": epochs,
        "total_combinations": total_runs
    }
    
    with open(summary_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    
    # Run experiments for each combination
    for i, (lr, batch_size) in enumerate(all_combinations):
        print(f"\n[{i+1}/{total_runs}] Running experiment with lr={lr}, batch_size={batch_size}")
        
        # Create a unique name for this run
        run_name = f"unet-lr{lr}-bs{batch_size}"
        
        # Create subdirectory for this experiment
        run_dir = experiment_dir / f"run_{i+1}_lr{lr}_bs{batch_size}"
        run_dir.mkdir(exist_ok=True)
        
        # Create output and log directories for this run
        output_dir = run_dir / "output"
        log_dir = run_dir / "logs"
        output_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        
        # Update config with current hyperparameters
        current_config = base_config.copy()
        current_config.update({
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "run_name": run_name,
            "use_wandb": True,
            "wandb_project": wandb_project
        })
        
        # Save current config
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(current_config, f)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Run training
            best_dice = train(current_config)
            
            # Calculate training time
            training_time = (time.time() - start_time) / 60  # minutes
            
            # Load metrics from the latest checkpoint
            checkpoint_path = output_dir / "checkpoint_best.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                val_dice = checkpoint.get('val_dice', 0.0)
                val_loss = checkpoint.get('val_loss', float('inf'))
                val_pixel_acc = checkpoint.get('val_pixel_acc', 0.0)
            else:
                val_dice = 0.0
                val_loss = float('inf')
                val_pixel_acc = 0.0
            
            print(f"Experiment completed - val_dice: {val_dice:.4f}, val_loss: {val_loss:.4f}, val_pixel_acc: {val_pixel_acc:.4f}")
            
            # Store results for this run
            result_row = [run_name, lr, batch_size, val_dice, val_loss, val_pixel_acc, training_time]
            all_results.append(result_row)
            
            # Save metrics to YAML file
            run_metrics = {
                "learning_rate": float(lr),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "val_dice": float(val_dice),
                "val_loss": float(val_loss),
                "val_pixel_acc": float(val_pixel_acc),
                "training_time_minutes": float(training_time)
            }
            with open(run_dir / "metrics.yaml", "w") as f:
                yaml.safe_dump(run_metrics, f)
                
        except Exception as e:
            print(f"Error in run {run_name}: {e}")
            tb.print_exc()
            result_row = [run_name, lr, batch_size, 0, float('inf'), 0, 0]
            all_results.append(result_row)
            
            # Save error information
            with open(run_dir / "error.txt", "w") as f:
                f.write(f"Error running experiment with lr={lr}, batch_size={batch_size}:\n")
                f.write(str(e))
                f.write("\n\nTraceback:\n")
                tb.print_exc(file=f)
    
    # Create summary plots and analysis
    print("\nCreating summary analysis...")
    
    try:
        # Convert results to DataFrame
        results_df = pd.DataFrame(
            all_results,
            columns=["Run Name", "Learning Rate", "Batch Size", "Val Dice", "Val Loss", "Val Pixel Acc", "Training Time (min)"]
        )
        
        # Save results to CSV
        csv_path = summary_dir / "results.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Create plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Plot learning rate vs. metrics
        for bs in sorted(set(results_df["Batch Size"])):
            mask = results_df["Batch Size"] == bs
            if any(mask):
                axes[0,0].plot(
                    results_df[mask]["Learning Rate"],
                    results_df[mask]["Val Dice"],
                    'o-',
                    label=f'BS={bs}'
                )
        
        axes[0,0].set_xlabel('Learning Rate')
        axes[0,0].set_ylabel('Validation Dice')
        axes[0,0].set_title('Learning Rate vs. Validation Dice')
        axes[0,0].set_xscale('log')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Plot batch size vs. metrics
        for lr in sorted(set(results_df["Learning Rate"])):
            mask = results_df["Learning Rate"] == lr
            if any(mask):
                axes[0,1].plot(
                    results_df[mask]["Batch Size"],
                    results_df[mask]["Val Dice"],
                    'o-',
                    label=f'LR={lr}'
                )
        
        axes[0,1].set_xlabel('Batch Size')
        axes[0,1].set_ylabel('Validation Dice')
        axes[0,1].set_title('Batch Size vs. Validation Dice')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Plot learning rate vs. pixel accuracy
        for bs in sorted(set(results_df["Batch Size"])):
            mask = results_df["Batch Size"] == bs
            if any(mask):
                axes[1,0].plot(
                    results_df[mask]["Learning Rate"],
                    results_df[mask]["Val Pixel Acc"],
                    'o-',
                    label=f'BS={bs}'
                )
        
        axes[1,0].set_xlabel('Learning Rate')
        axes[1,0].set_ylabel('Validation Pixel Accuracy')
        axes[1,0].set_title('Learning Rate vs. Validation Pixel Accuracy')
        axes[1,0].set_xscale('log')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Plot batch size vs. pixel accuracy
        for lr in sorted(set(results_df["Learning Rate"])):
            mask = results_df["Learning Rate"] == lr
            if any(mask):
                axes[1,1].plot(
                    results_df[mask]["Batch Size"],
                    results_df[mask]["Val Pixel Acc"],
                    'o-',
                    label=f'LR={lr}'
                )
        
        axes[1,1].set_xlabel('Batch Size')
        axes[1,1].set_ylabel('Validation Pixel Accuracy')
        axes[1,1].set_title('Batch Size vs. Validation Pixel Accuracy')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # Plot training time vs. batch size
        axes[2,0].plot(results_df["Batch Size"], results_df["Training Time (min)"], 'o-')
        axes[2,0].set_xlabel('Batch Size')
        axes[2,0].set_ylabel('Training Time (min)')
        axes[2,0].set_title('Training Time vs. Batch Size')
        axes[2,0].grid(True)
        
        # Plot validation loss vs. learning rate
        for bs in sorted(set(results_df["Batch Size"])):
            mask = results_df["Batch Size"] == bs
            if any(mask):
                axes[2,1].plot(
                    results_df[mask]["Learning Rate"],
                    results_df[mask]["Val Loss"],
                    'o-',
                    label=f'BS={bs}'
                )
        
        axes[2,1].set_xlabel('Learning Rate')
        axes[2,1].set_ylabel('Validation Loss')
        axes[2,1].set_title('Learning Rate vs. Validation Loss')
        axes[2,1].set_xscale('log')
        axes[2,1].legend()
        axes[2,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(summary_dir / "hyperparameter_tuning_results.png")
        
        # Create best results markdown file
        best_dice_idx = results_df["Val Dice"].idxmax()
        best_loss_idx = results_df["Val Loss"].idxmin()
        best_pixel_acc_idx = results_df["Val Pixel Acc"].idxmax()
        
        with open(summary_dir / "best_results.md", "w") as f:
            f.write(f"# Hyperparameter Tuning Results\n\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Date: {timestamp}\n\n")
            
            f.write(f"## Best Validation Dice\n\n")
            f.write(f"- Value: {results_df.iloc[best_dice_idx]['Val Dice']:.4f}\n")
            f.write(f"- Learning Rate: {results_df.iloc[best_dice_idx]['Learning Rate']}\n")
            f.write(f"- Batch Size: {results_df.iloc[best_dice_idx]['Batch Size']}\n\n")
            
            f.write(f"## Best Validation Loss\n\n")
            f.write(f"- Value: {results_df.iloc[best_loss_idx]['Val Loss']:.4f}\n")
            f.write(f"- Learning Rate: {results_df.iloc[best_loss_idx]['Learning Rate']}\n")
            f.write(f"- Batch Size: {results_df.iloc[best_loss_idx]['Batch Size']}\n\n")
            
            f.write(f"## Best Validation Pixel Accuracy\n\n")
            f.write(f"- Value: {results_df.iloc[best_pixel_acc_idx]['Val Pixel Acc']:.4f}\n")
            f.write(f"- Learning Rate: {results_df.iloc[best_pixel_acc_idx]['Learning Rate']}\n")
            f.write(f"- Batch Size: {results_df.iloc[best_pixel_acc_idx]['Batch Size']}\n\n")
            
            f.write(f"## All Results\n\n")
            f.write(results_df.to_markdown(index=False))
        
        # Log results to W&B
        if wandb.run is not None:
            wandb.log({
                "hyperparameter_tuning_results": wandb.Image(str(summary_dir / "hyperparameter_tuning_results.png")),
                "results_table": wandb.Table(dataframe=results_df)
            })
            # Log the CSV file as an artifact
            artifact = wandb.Artifact(
                name=f"hyperparameter_tuning_results_{timestamp}",
                type="results",
                description="Results of hyperparameter tuning experiments"
            )
            artifact.add_file(str(csv_path))
            wandb.log_artifact(artifact)
        
    except Exception as e:
        print(f"Error creating summary analysis: {e}")
        tb.print_exc()
    
    print("\nHyperparameter tuning complete!")
    print(f"Results saved to: {experiment_dir}")
    
    # Return the experiment directory path
    return experiment_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for UNet segmentation")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to base config YAML file")
    parser.add_argument("--lr", type=float, nargs='+', default=[1e-3, 5e-4, 1e-4],
                        help="Learning rates to try")
    parser.add_argument("--batch", type=int, nargs='+', default=[8, 16, 32],
                        help="Batch sizes to try")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs for each run")
    parser.add_argument("--wandb_project", type=str, default="unet-hyperparameter-tuning",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (username or team name)")
    
    args = parser.parse_args()
    
    exp_dir = run_hyperparameter_tuning(
        config_path=args.config,
        learning_rates=args.lr,
        batch_sizes=args.batch,
        epochs=args.epochs,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )
    
    print(f"\nResults saved to: {exp_dir}") 