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

# Import the training function from the existing train.py
from segmentation.ultralytics.train import train_multiclass_segmentation

def run_hyperparameter_tuning(
    dataset_yaml,
    learning_rates=[1e-3, 5e-4, 1e-4],
    batch_sizes=[8, 16, 32],
    epochs=20,
    imgsz=640,
    model_size='n',
    patience=5,
    device='0',
    wandb_project="segmentation-hyperparameter-tuning",
    wandb_entity=None
):
    """
    Run multiple training experiments with different hyperparameters.
    
    Args:
        dataset_yaml: Path to dataset YAML file
        learning_rates: List of learning rates to try
        batch_sizes: List of batch sizes to try
        epochs: Number of training epochs for each run
        imgsz: Image size for training
        model_size: Model size (n, s, m, l, x)
        patience: Early stopping patience
        device: Device to train on ('cpu', '0', '0,1,2,3', etc.)
        wandb_project: W&B project name
        wandb_entity: W&B entity (username or team name)
    """
    # Create experiment directory with unique name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"hyperparameter_tuning_{model_size}_{timestamp}"
    
    # Create experiment root directory inside runs/
    experiment_dir = Path("runs") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created experiment directory: {experiment_dir}")
    
    # Create summary subdirectory for plots and metrics
    summary_dir = experiment_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    # Verify wandb is working
    try:
        # Check if wandb is logged in
        if not wandb.api.api_key:
            print("W&B API key not found. Please log in with 'wandb login'")
            login_status = os.system("wandb login")
            if login_status != 0:
                print("Failed to log in to W&B. Continuing without W&B logging.")
                return
        
        print(f"W&B authenticated - ready to log experiments")
    except Exception as e:
        print(f"Error checking W&B authentication: {e}")
        print("Continuing but W&B logging may not work correctly")
    
    # Create a master W&B run to group all experiments
    master_run_name = f"tuning-{model_size}-{timestamp}"
    
    # Store results for creating a summary at the end
    all_results = []
    
    # Get dataset info
    try:
        with open(dataset_yaml, 'r') as f:
            dataset_info = yaml.safe_load(f)
            dataset_name = os.path.basename(dataset_info.get('path', 'dataset'))
    except:
        dataset_name = 'dataset'
    
    # Generate all combinations of hyperparameters
    all_combinations = list(itertools.product(learning_rates, batch_sizes))
    total_runs = len(all_combinations)
    
    print(f"Starting hyperparameter tuning with {total_runs} combinations")
    print(f"Master run name: {master_run_name}")
    print(f"W&B project: {wandb_project}")
    
    # Save hyperparameter config to experiment directory
    config = {
        "timestamp": timestamp,
        "dataset": dataset_yaml,
        "learning_rates": learning_rates,
        "batch_sizes": batch_sizes,
        "epochs": epochs,
        "model_size": model_size,
        "patience": patience,
        "device": device,
        "wandb_project": wandb_project,
        "total_combinations": total_runs
    }
    
    with open(summary_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    
    # Run experiments for each combination
    for i, (lr, batch_size) in enumerate(all_combinations):
        print(f"\n[{i+1}/{total_runs}] Running experiment with lr={lr}, batch_size={batch_size}, model={model_size}")
        
        # Create a unique name for this run that includes model size
        run_name = f"{dataset_name}-{model_size}-lr{lr}-bs{batch_size}"
        
        # Create subdirectory for this experiment that includes model size
        run_dir = experiment_dir / f"run_{i+1}_{model_size}_lr{lr}_bs{batch_size}"
        run_dir.mkdir(exist_ok=True)
        
        # Set environment variables to ensure W&B is enabled for the training run
        os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_entity:
            os.environ["WANDB_ENTITY"] = wandb_entity
        os.environ["WANDB_NAME"] = run_name
        # Redirect W&B output directory to our experiment subdirectory
        os.environ["WANDB_DIR"] = str(run_dir)
        
        print(f"Starting experiment with W&B name: {run_name}")
        print(f"W&B project: {wandb_project}")
        print(f"Experiment output directory: {run_dir}")
        if wandb_entity:
            print(f"W&B entity: {wandb_entity}")
        
        # Start timing
        start_time = time.time()
        
        # Run the training with these hyperparameters
        try:
            # Use the custom run dir for saving YOLOv8 outputs
            results, metrics = train_multiclass_segmentation(
                dataset_yaml=dataset_yaml,
                epochs=epochs,
                batch_size=batch_size,
                imgsz=imgsz,
                model_size=model_size,
                patience=patience,
                device=device,
                use_wandb=True,
                wandb_project=wandb_project,
                wandb_name=run_name,
                wandb_entity=wandb_entity,
                lr=lr,
                resume_wandb=False
            )
            
            # Calculate training time
            training_time = (time.time() - start_time) / 60  # minutes
            
            # Extract metrics and convert numpy types to native Python types
            box_map50 = getattr(metrics.box, 'map50', 0)
            if isinstance(box_map50, np.floating):
                box_map50 = float(box_map50)
                
            mask_map50 = getattr(metrics.seg, 'map50', 0)
            if isinstance(mask_map50, np.floating):
                mask_map50 = float(mask_map50)
            
            print(f"Experiment completed - box_mAP50: {box_map50:.4f}, mask_mAP50: {mask_map50:.4f}")
            
            # Store results for this run
            result_row = [run_name, lr, batch_size, box_map50, mask_map50, training_time]
            all_results.append(result_row)
            
            # Copy important artifacts to our run directory if they're not already there
            if results is not None and hasattr(results, 'save_dir'):
                yolo_save_dir = Path(results.save_dir)
                if yolo_save_dir.exists() and yolo_save_dir != run_dir:
                    # Copy weights, plots, and other artifacts
                    for artifact_dir in ['weights', 'plots']:
                        src_dir = yolo_save_dir / artifact_dir
                        if src_dir.exists():
                            dst_dir = run_dir / artifact_dir
                            dst_dir.mkdir(exist_ok=True)
                            for file in src_dir.glob('*'):
                                if not (dst_dir / file.name).exists():
                                    shutil.copy2(file, dst_dir / file.name)
            
            # Save metrics to YAML file in this run's directory
            run_metrics = {
                "learning_rate": float(lr),
                "batch_size": int(batch_size),
                "model_size": model_size,
                "epochs": int(epochs),
                "box_map50": float(box_map50),
                "mask_map50": float(mask_map50),
                "training_time_minutes": float(training_time)
            }
            with open(run_dir / "metrics.yaml", "w") as f:
                yaml.safe_dump(run_metrics, f)
                
        except Exception as e:
            print(f"Error in run {run_name}: {e}")
            tb.print_exc()
            result_row = [run_name, lr, batch_size, 0, 0, 0]
            all_results.append(result_row)
            
            # Save error information
            with open(run_dir / "error.txt", "w") as f:
                f.write(f"Error running experiment with lr={lr}, batch_size={batch_size}, model={model_size}:\n")
                f.write(str(e))
                f.write("\n\nTraceback:\n")
                tb.print_exc(file=f)
    
    # Create a summary run after all experiments are done
    print("\nCreating summary run...")
    
    try:
        # Initialize a new wandb run for the summary
        os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_entity:
            os.environ["WANDB_ENTITY"] = wandb_entity
        os.environ["WANDB_NAME"] = f"{master_run_name}-summary"
        os.environ["WANDB_DIR"] = str(summary_dir)
        
        wandb.init(
            project=wandb_project,
            name=f"{master_run_name}-summary",
            config={
                "learning_rates": learning_rates,
                "batch_sizes": batch_sizes,
                "epochs": epochs,
                "imgsz": imgsz,
                "model_size": model_size,
                "total_runs": total_runs,
                "completed_runs": len(all_results)
            },
            job_type="hyperparameter-summary"
        )
        
        print(f"Summary run created: {wandb.run.name} (ID: {wandb.run.id})")
        
        # Create a new table for this summary run
        summary_table = wandb.Table(
            columns=["Run Name", "Learning Rate", "Batch Size", "Box mAP50", "Mask mAP50", "Training Time (min)"]
        )
        
        # Add data to the table
        for row in all_results:
            summary_table.add_data(*row)
        
        # Log the table
        wandb.log({"results": summary_table})
        
        # Create summary plots
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Convert to numpy arrays for easier manipulation
            lrs = np.array([row[1] for row in all_results])
            batch_sizes = np.array([row[2] for row in all_results])
            box_maps = np.array([row[3] for row in all_results])
            mask_maps = np.array([row[4] for row in all_results])
            
            # Create plots
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot learning rate vs. mAP
            unique_bs = set(batch_sizes)
            for bs in unique_bs:
                mask = batch_sizes == bs
                if any(mask):  # Only plot if we have data for this batch size
                    axes[0].plot(lrs[mask], box_maps[mask], 'o-', label=f'BS={bs} (Box)')
                    axes[0].plot(lrs[mask], mask_maps[mask], 's--', label=f'BS={bs} (Mask)')
            
            axes[0].set_xlabel('Learning Rate')
            axes[0].set_ylabel('mAP50')
            axes[0].set_title('Learning Rate vs. mAP50')
            axes[0].set_xscale('log')
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot batch size vs. mAP
            unique_lrs = set(lrs)
            for lr in unique_lrs:
                mask = lrs == lr
                if any(mask):  # Only plot if we have data for this learning rate
                    axes[1].plot(batch_sizes[mask], box_maps[mask], 'o-', label=f'LR={lr} (Box)')
                    axes[1].plot(batch_sizes[mask], mask_maps[mask], 's--', label=f'LR={lr} (Mask)')
            
            axes[1].set_xlabel('Batch Size')
            axes[1].set_ylabel('mAP50')
            axes[1].set_title('Batch Size vs. mAP50')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            
            # Save the figure and log to wandb
            fig_path = summary_dir / f"hyperparameter_tuning_results.png"
            plt.savefig(fig_path)
            wandb.log({"summary_plots": wandb.Image(str(fig_path))})
            
            # Create additional plots showing learning rate and batch size impact
            # Box mAP by parameter
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Sort results by parameters
            sorted_lrs = sorted(list(unique_lrs))
            lr_data = {lr: [] for lr in sorted_lrs}
            for row in all_results:
                lr_data[row[1]].append(row[3])  # box_map
                
            sorted_bs = sorted(list(unique_bs))
            bs_data = {bs: [] for bs in sorted_bs}
            for row in all_results:
                bs_data[row[2]].append(row[3])  # box_map
            
            # Box plots
            axes[0].boxplot([lr_data[lr] for lr in sorted_lrs], labels=[f"{lr}" for lr in sorted_lrs])
            axes[0].set_title("Box mAP50 by Learning Rate")
            axes[0].set_xlabel("Learning Rate")
            axes[0].set_ylabel("Box mAP50")
            axes[0].grid(True, linestyle='--', alpha=0.7)
            
            axes[1].boxplot([bs_data[bs] for bs in sorted_bs], labels=[f"{bs}" for bs in sorted_bs])
            axes[1].set_title("Box mAP50 by Batch Size")
            axes[1].set_xlabel("Batch Size")
            axes[1].set_ylabel("Box mAP50")
            axes[1].grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            fig_path = summary_dir / f"parameter_effects_box_map.png"
            plt.savefig(fig_path)
            wandb.log({"parameter_effects_box": wandb.Image(str(fig_path))})
            
            # Same for mask mAP
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Sort results by parameters
            lr_data = {lr: [] for lr in sorted_lrs}
            for row in all_results:
                lr_data[row[1]].append(row[4])  # mask_map
                
            bs_data = {bs: [] for bs in sorted_bs}
            for row in all_results:
                bs_data[row[2]].append(row[4])  # mask_map
            
            # Box plots
            axes[0].boxplot([lr_data[lr] for lr in sorted_lrs], labels=[f"{lr}" for lr in sorted_lrs])
            axes[0].set_title("Mask mAP50 by Learning Rate")
            axes[0].set_xlabel("Learning Rate")
            axes[0].set_ylabel("Mask mAP50")
            axes[0].grid(True, linestyle='--', alpha=0.7)
            
            axes[1].boxplot([bs_data[bs] for bs in sorted_bs], labels=[f"{bs}" for bs in sorted_bs])
            axes[1].set_title("Mask mAP50 by Batch Size")
            axes[1].set_xlabel("Batch Size")
            axes[1].set_ylabel("Mask mAP50")
            axes[1].grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            fig_path = summary_dir / f"parameter_effects_mask_map.png"
            plt.savefig(fig_path)
            wandb.log({"parameter_effects_mask": wandb.Image(str(fig_path))})
            
            # Save results to CSV
            import pandas as pd
            results_df = pd.DataFrame(all_results, 
                                     columns=["Run Name", "Learning Rate", "Batch Size", 
                                             "Box mAP50", "Mask mAP50", "Training Time (min)"])
            csv_path = summary_dir / "results.csv"
            results_df.to_csv(csv_path, index=False)
            
            # Create a best results markdown file
            best_box_idx = results_df["Box mAP50"].idxmax()
            best_mask_idx = results_df["Mask mAP50"].idxmax()
            
            with open(summary_dir / "best_results.md", "w") as f:
                f.write(f"# Hyperparameter Tuning Results\n\n")
                f.write(f"Experiment: {experiment_name}\n")
                f.write(f"Model: YOLOv8{model_size}-seg\n")
                f.write(f"Dataset: {dataset_yaml}\n")
                f.write(f"Date: {timestamp}\n\n")
                
                f.write(f"## Best Box mAP50\n\n")
                f.write(f"- Value: {results_df.iloc[best_box_idx]['Box mAP50']:.4f}\n")
                f.write(f"- Learning Rate: {results_df.iloc[best_box_idx]['Learning Rate']}\n")
                f.write(f"- Batch Size: {results_df.iloc[best_box_idx]['Batch Size']}\n\n")
                
                f.write(f"## Best Mask mAP50\n\n")
                f.write(f"- Value: {results_df.iloc[best_mask_idx]['Mask mAP50']:.4f}\n")
                f.write(f"- Learning Rate: {results_df.iloc[best_mask_idx]['Learning Rate']}\n")
                f.write(f"- Batch Size: {results_df.iloc[best_mask_idx]['Batch Size']}\n\n")
                
                f.write(f"## All Results\n\n")
                f.write(results_df.to_markdown(index=False))
            
        except Exception as e:
            print(f"Error creating summary plots: {e}")
            tb.print_exc()
        
        # Finish the wandb run
        wandb.finish()
        
    except Exception as e:
        print(f"Error in final summary: {e}")
        tb.print_exc()
    
    print("\nHyperparameter tuning complete!")
    print(f"Results saved to: {experiment_dir}")
    print(f"Results logged to W&B project: {wandb_project}")
    print(f"Check your experiments at: https://wandb.ai/{wandb_entity or 'user'}/{wandb_project}")
    
    # Return the experiment directory path so the user can easily locate results
    return experiment_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for YOLOv8 segmentation")
    parser.add_argument("--data", type=str, required=True, 
                        help="Path to dataset YAML file")
    parser.add_argument("--lr", type=float, nargs='+', default=[1e-3, 5e-4, 1e-4],
                        help="Learning rates to try")
    parser.add_argument("--batch", type=int, nargs='+', default=[8, 16, 32],
                        help="Batch sizes to try")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs for each run")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--model_size", type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--device", type=str, default='',
                        help="Device to train on ('cpu', '0', '0,1,2,3', etc.)")
    parser.add_argument("--wandb_project", type=str, default="segmentation-hyperparameter-tuning",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (username or team name)")
    
    args = parser.parse_args()
    
    exp_dir = run_hyperparameter_tuning(
        dataset_yaml=args.data,
        learning_rates=args.lr,
        batch_sizes=args.batch,
        epochs=args.epochs,
        imgsz=args.imgsz,
        model_size=args.model_size,
        patience=args.patience,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )
    
    print(f"\nResults saved to: {exp_dir}") 