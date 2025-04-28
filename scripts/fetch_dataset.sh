#!/bin/bash

# Function to print colored text
print_colored() {
    local color=$1
    local text=$2
    
    case $color in
        "red")
            echo -e "\033[0;31m$text\033[0m"
            ;;
        "green")
            echo -e "\033[0;32m$text\033[0m"
            ;;
        "yellow")
            echo -e "\033[0;33m$text\033[0m"
            ;;
        "blue")
            echo -e "\033[0;34m$text\033[0m"
            ;;
        *)
            echo "$text"
            ;;
    esac
}

# Store the original working directory
ORIGINAL_DIR=$(pwd)
DATASET_BASE_DIR="$ORIGINAL_DIR/datasets"

# Ask for the dataset name
read -p "Enter a name for the dataset: " DATASET_NAME

# Create dataset directory
mkdir -p "$DATASET_BASE_DIR/$DATASET_NAME"
mkdir -p "$DATASET_BASE_DIR/$DATASET_NAME/images/train"
mkdir -p "$DATASET_BASE_DIR/$DATASET_NAME/images/val"
mkdir -p "$DATASET_BASE_DIR/$DATASET_NAME/images/test"
mkdir -p "$DATASET_BASE_DIR/$DATASET_NAME/labels/train"
mkdir -p "$DATASET_BASE_DIR/$DATASET_NAME/labels/val"
mkdir -p "$DATASET_BASE_DIR/$DATASET_NAME/labels/test"

# Create a temporary directory for download
TEMP_DIR="$ORIGINAL_DIR/temp_download"
mkdir -p $TEMP_DIR
cd $TEMP_DIR

# Ask for download URL
read -p "Enter the URL to download the dataset from: " DOWNLOAD_URL

# Download the dataset
print_colored "blue" "Downloading dataset from $DOWNLOAD_URL..."
curl -L $DOWNLOAD_URL -o dataset.zip

# Unzip the dataset
print_colored "blue" "Extracting dataset..."
unzip -q dataset.zip

# Assuming the unzipped structure has train/, valid/, test/ directories
# and data.yaml in the root of the extracted directory

# Find the root directory of the extracted dataset (might be in a subdirectory)
ROOT_DIR=$(find . -type f -name "data.yaml" -exec dirname {} \; | head -1)

if [ -z "$ROOT_DIR" ]; then
    print_colored "red" "Error: Could not find data.yaml in the extracted dataset."
    cd $ORIGINAL_DIR
    rm -rf $TEMP_DIR
    exit 1
fi

cd "$ROOT_DIR"

# Move files to the YOLO format directory structure
print_colored "blue" "Reorganizing dataset into YOLO format..."

# Training set
if [ -d "train/images" ]; then
    cp -r train/images/* "$DATASET_BASE_DIR/$DATASET_NAME/images/train/"
    print_colored "green" "Copied training images."
else
    print_colored "yellow" "Warning: No training images found."
fi

if [ -d "train/labels" ]; then
    cp -r train/labels/* "$DATASET_BASE_DIR/$DATASET_NAME/labels/train/"
    print_colored "green" "Copied training labels."
else
    print_colored "yellow" "Warning: No training labels found."
fi

# Validation set (could be named 'valid' or 'val')
if [ -d "valid/images" ]; then
    cp -r valid/images/* "$DATASET_BASE_DIR/$DATASET_NAME/images/val/"
    print_colored "green" "Copied validation images."
elif [ -d "val/images" ]; then
    cp -r val/images/* "$DATASET_BASE_DIR/$DATASET_NAME/images/val/"
    print_colored "green" "Copied validation images."
else
    print_colored "yellow" "Warning: No validation images found."
fi

if [ -d "valid/labels" ]; then
    cp -r valid/labels/* "$DATASET_BASE_DIR/$DATASET_NAME/labels/val/"
    print_colored "green" "Copied validation labels."
elif [ -d "val/labels" ]; then
    cp -r val/labels/* "$DATASET_BASE_DIR/$DATASET_NAME/labels/val/"
    print_colored "green" "Copied validation labels."
else
    print_colored "yellow" "Warning: No validation labels found."
fi

# Test set
if [ -d "test/images" ]; then
    cp -r test/images/* "$DATASET_BASE_DIR/$DATASET_NAME/images/test/"
    print_colored "green" "Copied test images."
else
    print_colored "yellow" "Warning: No test images found."
fi

if [ -d "test/labels" ]; then
    cp -r test/labels/* "$DATASET_BASE_DIR/$DATASET_NAME/labels/test/"
    print_colored "green" "Copied test labels."
else
    print_colored "yellow" "Warning: No test labels found."
fi

# Process the data.yaml file
if [ -f "data.yaml" ]; then
    # Copy the original yaml file
    cp data.yaml "$DATASET_BASE_DIR/$DATASET_NAME/dataset.yaml"
    
    # Update paths in the yaml file
    cd "$DATASET_BASE_DIR/$DATASET_NAME/"
    sed -i "s|path:.*|path: \"$DATASET_BASE_DIR/$DATASET_NAME\"|g" dataset.yaml
    sed -i "s|train:.*|train: images/train|g" dataset.yaml
    sed -i "s|val:.*|val: images/val|g" dataset.yaml
    sed -i "s|test:.*|test: images/test|g" dataset.yaml
    
    print_colored "green" "Created and updated dataset.yaml"
else
    print_colored "red" "Error: data.yaml not found in the extracted dataset."
fi

# Clean up
cd $ORIGINAL_DIR
rm -rf $TEMP_DIR

print_colored "green" "Dataset setup complete!"
print_colored "blue" "Dataset location: datasets/$DATASET_NAME"
print_colored "yellow" "You can now use this dataset for training with:"
echo "yolo segment train data=datasets/$DATASET_NAME/dataset.yaml"

# Run the dataset info script to show stats
if [ -f "get_dataset_info.sh" ]; then
    print_colored "blue" "Running dataset info script to show statistics:"
    ./get_dataset_info.sh
fi 