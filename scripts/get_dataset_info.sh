#!/bin/bash

# Script to provide detailed information about available datasets

set -e

# Function to print colored text
print_colored() {
    local color=$1
    local text=$2
    case $color in
        "green") echo -e "\033[0;32m$text\033[0m" ;;
        "blue") echo -e "\033[0;34m$text\033[0m" ;;
        "yellow") echo -e "\033[0;33m$text\033[0m" ;;
        "red") echo -e "\033[0;31m$text\033[0m" ;;
        *) echo "$text" ;;
    esac
}

# Function to count files in a directory
count_files() {
    if [ -d "$1" ]; then
        find "$1" -type f | wc -l
    else
        echo "0"
    fi
}

# Check if datasets directory exists
if [ ! -d "datasets" ]; then
    print_colored "red" "Error: 'datasets' directory not found."
    print_colored "yellow" "Please run setup_dataset.sh first to create a dataset."
    exit 1
fi

# List all datasets
datasets=$(find datasets -maxdepth 1 -mindepth 1 -type d)

if [ -z "$datasets" ]; then
    print_colored "yellow" "No datasets found in the 'datasets' directory."
    print_colored "blue" "Run setup_dataset.sh to create a dataset."
    exit 0
fi

print_colored "green" "=== Available Datasets ==="
echo ""

# Process each dataset
for dataset_path in $datasets; do
    dataset_name=$(basename "$dataset_path")
    print_colored "blue" "Dataset: $dataset_name"
    
    # Check for dataset.yaml
    if [ -f "$dataset_path/dataset.yaml" ]; then
        echo "Configuration: $dataset_path/dataset.yaml"
        
        # Extract class information
        if grep -q "names:" "$dataset_path/dataset.yaml"; then
            echo "Classes: $(grep "names:" "$dataset_path/dataset.yaml" | sed 's/names: //')"
        fi
        
        if grep -q "nc:" "$dataset_path/dataset.yaml"; then
            echo "Number of classes: $(grep "nc:" "$dataset_path/dataset.yaml" | sed 's/nc: //')"
        fi
    else
        print_colored "yellow" "Warning: No dataset.yaml found"
    fi
    
    # Count images and labels
    echo "Statistics:"
    
    # Training set
    train_img_count=$(count_files "$dataset_path/images/train")
    train_label_count=$(count_files "$dataset_path/labels/train")
    echo "  Training:   $train_img_count images, $train_label_count labels"
    
    # Validation set
    val_img_count=$(count_files "$dataset_path/images/val")
    val_label_count=$(count_files "$dataset_path/labels/val")
    echo "  Validation: $val_img_count images, $val_label_count labels"
    
    # Test set
    test_img_count=$(count_files "$dataset_path/images/test")
    test_label_count=$(count_files "$dataset_path/labels/test")
    echo "  Testing:    $test_img_count images, $test_label_count labels"
    
    # Total
    total_img=$((train_img_count + val_img_count + test_img_count))
    total_label=$((train_label_count + val_label_count + test_label_count))
    echo "  Total:      $total_img images, $total_label labels"
    
    # Check for any issues
    if [ $train_img_count -ne $train_label_count ]; then
        print_colored "yellow" "  Warning: Number of training images ($train_img_count) does not match labels ($train_label_count)"
    fi
    
    if [ $val_img_count -ne $val_label_count ]; then
        print_colored "yellow" "  Warning: Number of validation images ($val_img_count) does not match labels ($val_label_count)"
    fi
    
    if [ $test_img_count -ne $test_label_count ]; then
        print_colored "yellow" "  Warning: Number of test images ($test_img_count) does not match labels ($test_label_count)"
    fi
    
    echo ""
done

print_colored "green" "=== Usage Instructions ==="
echo "To use a dataset for training:"
echo "python segmentation/ultralytics/train.py --dataset datasets/DATASET_NAME/dataset.yaml --epochs 20 --batch_size 16 --model_size n"
echo ""
echo "To visualize a dataset:"
echo "python segmentation/ultralytics/visualize_yolo_dataset.py --data datasets/DATASET_NAME/dataset.yaml" 