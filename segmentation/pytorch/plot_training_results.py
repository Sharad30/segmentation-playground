import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def load_results(directory):
    """Load results.csv from a directory with validation."""
    try:
        df = pd.read_csv(f"{directory}/results.csv")
        
        # Print basic statistics for debugging
        print(f"\nAnalyzing {directory}:")
        print(f"Number of epochs: {len(df)}")
        
        # Check for NaN and infinite values
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32]:
                if df[col].isna().any() or np.isinf(df[col]).any():
                    print(f"Warning: Invalid values found in {col}")
                    print(f"NaN count: {df[col].isna().sum()}")
                    print(f"Inf count: {np.isinf(df[col]).sum()}")
        
        # Print statistics for each loss column
        loss_cols = [col for col in df.columns if 'loss' in col.lower()]
        print("\nLoss statistics:")
        for col in loss_cols:
            print(f"{col}:")
            print(f"  Min: {df[col].min():.4f}")
            print(f"  Max: {df[col].max():.4f}")
            print(f"  Mean: {df[col].mean():.4f}")
            print(f"  Last 5 values: {df[col].tail().values}")
        
        return df
    except Exception as e:
        print(f"Error loading {directory}: {str(e)}")
        return None

def get_best_model_stats(base_dir):
    """Get best model stats for each combination and save to CSV"""
    stats = []
    
    for lr in ['0.001', '0.0001', '1e-05']:
        for bs in [16, 32, 64]:
            dir_name = f"buildings-x-lr{lr}-bs{bs}"
            df = load_results(Path(base_dir) / dir_name)
            
            if df is not None:
                # Get the last row (final metrics)
                final_metrics = df.iloc[-1].to_dict()
                
                # Extract relevant metrics
                stats.append({
                    'learning_rate': lr,
                    'batch_size': bs,
                    'mAP50': final_metrics.get('metrics/mAP50(B)', 0),
                    'mAP50-95': final_metrics.get('metrics/mAP50-95(B)', 0),
                    'precision': final_metrics.get('metrics/precision(B)', 0),
                    'recall': final_metrics.get('metrics/recall(B)', 0),
                    'train_box_loss': final_metrics.get('train/box_loss', 0),
                    'train_cls_loss': final_metrics.get('train/cls_loss', 0),
                    'train_dfl_loss': final_metrics.get('train/dfl_loss', 0),
                    'train_seg_loss': final_metrics.get('train/seg_loss', 0),
                    'val_box_loss': final_metrics.get('val/box_loss', 0),
                    'val_cls_loss': final_metrics.get('val/cls_loss', 0),
                    'val_dfl_loss': final_metrics.get('val/dfl_loss', 0),
                    'val_seg_loss': final_metrics.get('val/seg_loss', 0)
                })
    
    # Convert to DataFrame and sort by mAP50
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('mAP50', ascending=False)
    
    # Save to CSV
    output_dir = Path('artifacts/pytorch/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    return stats_df

def plot_loss_comparison(base_dir, loss_type):
    """Plot side-by-side comparison of training and validation losses."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Define colors for batch sizes and line styles for learning rates
    batch_colors = {16: 'blue', 32: 'green', 64: 'red'}
    lr_styles = {0.001: '-', 0.0001: '--', 1e-05: ':'}
    
    # Keep track of plotted data to avoid duplicates
    plotted_data = set()
    
    for lr in [0.001, 0.0001, 1e-05]:
        for bs in [16, 32, 64]:
            dir_name = f"{base_dir}/buildings-x-lr{lr}-bs{bs}"
            print(f"\nProcessing directory: {dir_name}")
            
            df = load_results(dir_name)
            
            if df is not None:
                train_col = f'train/{loss_type}'
                val_col = f'val/{loss_type}'
                
                if train_col in df.columns and val_col in df.columns:
                    # Create a hash of the data to check for duplicates
                    data_hash = hash(tuple(df[train_col].tolist()) + tuple(df[val_col].tolist()))
                    
                    if data_hash in plotted_data:
                        print(f"WARNING: Duplicate data found for lr={lr}, bs={bs}")
                        continue
                    
                    plotted_data.add(data_hash)
                    print(f"Plotting new data for lr={lr}, bs={bs}")
                    
                    # Calculate moving averages
                    window_size = 3
                    train_ma = df[train_col].rolling(window=window_size, center=True).mean()
                    val_ma = df[val_col].rolling(window=window_size, center=True).mean()
                    
                    # Plot training loss
                    ax1.plot(df.index, train_ma, 
                            label=f'lr={lr}, bs={bs}',
                            color=batch_colors[bs],
                            linestyle=lr_styles[lr])
                    
                    # Plot validation loss
                    ax2.plot(df.index, val_ma, 
                            label=f'lr={lr}, bs={bs}',
                            color=batch_colors[bs],
                            linestyle=lr_styles[lr])
                    
                    print(f"Successfully plotted data for lr={lr}, bs={bs}")
                else:
                    print(f"Missing columns {train_col} or {val_col} for lr={lr}, bs={bs}")
    
    # Configure training plot
    ax1.set_title(f"Training {loss_type.replace('_', ' ').title()}")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Configure validation plot
    ax2.set_title(f"Validation {loss_type.replace('_', ' ').title()}")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add overall title
    fig.suptitle(f"{loss_type.replace('_', ' ').title()} - Training vs Validation", y=1.05, fontsize=16)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('artifacts/pytorch/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{loss_type}_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n✓ Generated side-by-side comparison plot for {loss_type}")

def main():
    base_dir = 'runs/train'
    
    print("Generating model comparison statistics...")
    stats_df = get_best_model_stats(base_dir)
    print("✓ Generated model comparison CSV")
    
    # Get best model parameters
    best_model = stats_df.iloc[0]
    best_lr = best_model['learning_rate']
    best_bs = best_model['batch_size']
    
    print(f"\nBest model configuration:")
    print(f"Learning Rate: {best_lr}")
    print(f"Batch Size: {best_bs}")
    print(f"mAP50: {best_model['mAP50']:.4f}")
    print(f"mAP50-95: {best_model['mAP50-95']:.4f}")
    
    # Plot side-by-side training and validation losses
    loss_types = ['box_loss', 'seg_loss', 'cls_loss', 'dfl_loss']
    for loss_type in loss_types:
        plot_loss_comparison(base_dir, loss_type)
    
    print("\nPlots have been saved to artifacts/pytorch/plots/")
    print("- model_comparison.csv: Comparison of all model configurations")
    print("- *_comparison.png: Side-by-side training vs validation loss plots")

if __name__ == '__main__':
    main() 