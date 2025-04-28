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

# Check for required Python packages
check_dependencies() {
    print_colored "blue" "Checking dependencies..."
    
    # List of required packages with their specific pip package names
    declare -A PACKAGE_MAP
    PACKAGE_MAP["numpy"]="numpy"
    PACKAGE_MAP["skimage"]="scikit-image"
    PACKAGE_MAP["shapely"]="shapely"
    PACKAGE_MAP["cv2"]="opencv-python"
    
    MISSING_PACKAGES=()
    MISSING_PACKAGE_NAMES=()
    
    # Check each dependency
    for package in "${!PACKAGE_MAP[@]}"; do
        pip_package="${PACKAGE_MAP[$package]}"
        
        if ! python3 -c "import $package" &>/dev/null; then
            MISSING_PACKAGES+=("$package")
            MISSING_PACKAGE_NAMES+=("$pip_package")
        fi
    done
    
    # Install missing packages one by one
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        print_colored "yellow" "Installing missing dependencies: ${MISSING_PACKAGE_NAMES[*]}"
        
        for i in "${!MISSING_PACKAGES[@]}"; do
            package="${MISSING_PACKAGES[$i]}"
            pip_package="${MISSING_PACKAGE_NAMES[$i]}"
            
            print_colored "blue" "Installing $pip_package..."
            if ! pip install "$pip_package" --no-cache-dir; then
                print_colored "yellow" "Trying alternative installation method for $pip_package..."
                if ! pip install --user "$pip_package"; then
                    print_colored "red" "Failed to install $pip_package. Please install it manually:"
                    print_colored "yellow" "pip install $pip_package"
                    
                    # Print more specific instructions for scikit-image
                    if [ "$pip_package" == "scikit-image" ]; then
                        print_colored "yellow" "For scikit-image, you might need to install dependencies first:"
                        print_colored "yellow" "pip install numpy scipy pillow networkx"
                        print_colored "yellow" "Then try: pip install scikit-image==0.19.3"
                    fi
                    
                    # Ask if user wants to continue without this dependency
                    read -p "Do you want to continue anyway? (y/n): " CONTINUE
                    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
                        exit 1
                    fi
                fi
            fi
            
            # Verify installation
            if ! python3 -c "import $package" &>/dev/null; then
                print_colored "red" "Warning: $package still cannot be imported."
            else
                print_colored "green" "Successfully installed $pip_package!"
            fi
        done
        
        print_colored "green" "Dependency installation complete!"
    else
        print_colored "green" "All dependencies are already installed."
    fi
}

# Function to count segmentations with non-empty lists
count_non_empty_segmentations() {
    local json_file=$1
    
    # Using Python to parse and count JSON properly
    python3 - <<EOF
import json

try:
    with open("$json_file", 'r') as f:
        data = json.load(f)
    
    total_annotations = len(data.get('annotations', []))
    non_empty_seg = 0
    
    for ann in data.get('annotations', []):
        seg = ann.get('segmentation', [])
        if seg and len(seg) > 0 and isinstance(seg[0], list) and len(seg[0]) > 0:
            non_empty_seg += 1
    
    print(f"Total annotations: {total_annotations}")
    print(f"With segmentation: {non_empty_seg}")
    
    if total_annotations > 0:
        percent = (non_empty_seg / total_annotations) * 100
        print(f"Percentage with segmentation: {percent:.2f}%")
    else:
        print("No annotations found")
    
except Exception as e:
    print(f"Error analyzing JSON: {str(e)}")
EOF
}

# Set paths
VOC_DIR="/home/ubuntu/sharad/segmentation-playground/datasets/raw/VOCdevkit/VOC2012"
PROCESSED_DIR="/home/ubuntu/sharad/segmentation-playground/datasets/processed"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PYTHON_SCRIPT="$SCRIPT_DIR/voc_to_coco.py"

# Ask user for dataset name
print_colored "blue" "Enter a name for the dataset (e.g., voc_coco_semantic):"
read -p "> " DATASET_NAME

# Validate dataset name
if [ -z "$DATASET_NAME" ]; then
    print_colored "yellow" "No name provided, using default name 'voc_coco'"
    DATASET_NAME="voc_coco"
fi

# Remove any spaces and special characters from dataset name
DATASET_NAME=$(echo "$DATASET_NAME" | tr -cd '[:alnum:]_-')

# Set output directory
OUTPUT_DIR="$PROCESSED_DIR/$DATASET_NAME"

# Check if VOC directory exists
if [ ! -d "$VOC_DIR" ]; then
    print_colored "red" "Error: VOC directory not found at $VOC_DIR"
    exit 1
fi

# Check if output directory already exists
if [ -d "$OUTPUT_DIR" ]; then
    print_colored "yellow" "Warning: Output directory already exists: $OUTPUT_DIR"
    read -p "Do you want to overwrite it? (y/n): " OVERWRITE
    if [ "$OVERWRITE" != "y" ] && [ "$OVERWRITE" != "Y" ]; then
        print_colored "blue" "Please run again with a different dataset name"
        exit 0
    else
        print_colored "yellow" "Overwriting existing directory..."
        rm -rf "$OUTPUT_DIR"
    fi
fi

# Check if SegmentationClass and SegmentationObject directories exist
SEGMENTATION_CLASS_DIR="$VOC_DIR/SegmentationClass"
SEGMENTATION_OBJECT_DIR="$VOC_DIR/SegmentationObject"

print_colored "blue" "Checking segmentation directories..."
if [ -d "$SEGMENTATION_CLASS_DIR" ]; then
    print_colored "green" "Found SegmentationClass directory: $SEGMENTATION_CLASS_DIR"
    CLASS_FILES_COUNT=$(find "$SEGMENTATION_CLASS_DIR" -type f -name "*.png" | wc -l)
    print_colored "blue" "Found $CLASS_FILES_COUNT segmentation class files"
else
    print_colored "yellow" "Warning: SegmentationClass directory not found at $SEGMENTATION_CLASS_DIR"
fi

if [ -d "$SEGMENTATION_OBJECT_DIR" ]; then
    print_colored "green" "Found SegmentationObject directory: $SEGMENTATION_OBJECT_DIR"
    OBJECT_FILES_COUNT=$(find "$SEGMENTATION_OBJECT_DIR" -type f -name "*.png" | wc -l)
    print_colored "blue" "Found $OBJECT_FILES_COUNT segmentation object files"
else
    print_colored "yellow" "Warning: SegmentationObject directory not found at $SEGMENTATION_OBJECT_DIR"
fi

# Check dependencies
check_dependencies

# Ensure the Python script is executable
chmod +x "$PYTHON_SCRIPT"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Ask for segmentation type
print_colored "blue" "Choose segmentation type:"
print_colored "yellow" "1) Instance segmentation (uses SegmentationObject folder)"
print_colored "yellow" "2) Semantic segmentation (uses SegmentationClass folder)"
read -p "Enter choice (1/2): " SEG_CHOICE

if [ "$SEG_CHOICE" == "2" ]; then
    SEG_TYPE="semantic"
    print_colored "blue" "Using semantic segmentation masks from SegmentationClass directory"
    if [ ! -d "$SEGMENTATION_CLASS_DIR" ]; then
        print_colored "red" "Warning: SegmentationClass directory not found, segmentation might fail"
    else
        print_colored "green" "Found SegmentationClass directory with $CLASS_FILES_COUNT mask files"
        print_colored "blue" "-------------------------------------------------------------------------"
        print_colored "blue" "SEMANTIC SEGMENTATION: Each pixel in the mask represents a class category."
        print_colored "blue" "All instances of the same class will have the same color in the mask."
        print_colored "blue" "Good for tasks where you don't need to distinguish between instances."
        print_colored "blue" "-------------------------------------------------------------------------"
    fi
    
    # Check for Segmentation imagesets directory
    SEGMENTATION_IMAGESETS_DIR="$VOC_DIR/ImageSets/Segmentation"
    if [ -d "$SEGMENTATION_IMAGESETS_DIR" ]; then
        print_colored "green" "Found ImageSets/Segmentation directory"
        TRAIN_COUNT=$(wc -l < "$SEGMENTATION_IMAGESETS_DIR/train.txt" 2>/dev/null || echo "0")
        VAL_COUNT=$(wc -l < "$SEGMENTATION_IMAGESETS_DIR/val.txt" 2>/dev/null || echo "0")
        print_colored "blue" "Found approximately $TRAIN_COUNT training and $VAL_COUNT validation images with segmentation masks"
        print_colored "blue" "The script will use only images with segmentation masks from ImageSets/Segmentation"
    else
        print_colored "yellow" "Warning: ImageSets/Segmentation directory not found. Will use ImageSets/Main instead."
        print_colored "yellow" "This might include images without segmentation masks, which will fall back to bounding boxes."
    fi
else
    SEG_TYPE="instance"
    print_colored "blue" "Using instance segmentation masks from SegmentationObject directory"
    if [ ! -d "$SEGMENTATION_OBJECT_DIR" ]; then
        print_colored "red" "Warning: SegmentationObject directory not found, segmentation might fail"
    else
        print_colored "green" "Found SegmentationObject directory with $OBJECT_FILES_COUNT mask files"
        print_colored "blue" "-------------------------------------------------------------------------"
        print_colored "blue" "INSTANCE SEGMENTATION: Each object instance has a unique ID in the mask."
        print_colored "blue" "Different instances of the same class will have different colors."
        print_colored "blue" "Good for tasks where you need to distinguish individual objects."
        print_colored "blue" "-------------------------------------------------------------------------"
    fi
    
    # Check for Segmentation imagesets directory
    SEGMENTATION_IMAGESETS_DIR="$VOC_DIR/ImageSets/Segmentation"
    if [ -d "$SEGMENTATION_IMAGESETS_DIR" ]; then
        print_colored "green" "Found ImageSets/Segmentation directory"
        TRAIN_COUNT=$(wc -l < "$SEGMENTATION_IMAGESETS_DIR/train.txt" 2>/dev/null || echo "0")
        VAL_COUNT=$(wc -l < "$SEGMENTATION_IMAGESETS_DIR/val.txt" 2>/dev/null || echo "0")
        print_colored "blue" "Found approximately $TRAIN_COUNT training and $VAL_COUNT validation images with segmentation masks"
        print_colored "blue" "The script will use only images with segmentation masks from ImageSets/Segmentation"
    else
        print_colored "yellow" "Warning: ImageSets/Segmentation directory not found. Will use ImageSets/Main instead."
        print_colored "yellow" "This might include images without segmentation masks, which will fall back to bounding boxes."
    fi
fi

# Ask for polygon simplification tolerance
print_colored "blue" "Set polygon simplification tolerance:"
print_colored "yellow" "Lower values (e.g., 0.1) create more detailed polygons with more points"
print_colored "yellow" "Higher values (e.g., 2.0) create simpler polygons with fewer points"
print_colored "yellow" "Default is 0.5"
read -p "Enter tolerance value [0.5]: " TOLERANCE

# Set default value if empty
if [ -z "$TOLERANCE" ]; then
    TOLERANCE="0.5"
fi

# Validate input is a number
if ! [[ "$TOLERANCE" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    print_colored "yellow" "Invalid input, using default tolerance of 0.5"
    TOLERANCE="0.5"
fi

print_colored "blue" "Starting VOC to COCO conversion with segmentation masks..."
print_colored "blue" "VOC directory: $VOC_DIR"
print_colored "blue" "Output directory: $OUTPUT_DIR"
print_colored "blue" "Dataset name: $DATASET_NAME"
print_colored "blue" "Segmentation type: $SEG_TYPE"
print_colored "blue" "Polygon simplification tolerance: $TOLERANCE"
print_colored "blue" "Converting dataset..."

# Run the Python script to convert VOC to COCO with full verbosity
python3 "$PYTHON_SCRIPT" \
    --voc-dir "$VOC_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --sets train val \
    --seg-type "$SEG_TYPE" \
    --simplify-tolerance "$TOLERANCE" \
    --verbose

# Check if conversion was successful
if [ $? -eq 0 ]; then
    print_colored "green" "Conversion completed successfully!"
    
    # Count images and annotations
    TRAIN_IMAGES=$(find "$OUTPUT_DIR/images/train" -type f | wc -l)
    VAL_IMAGES=$(find "$OUTPUT_DIR/images/val" -type f | wc -l)
    TEST_IMAGES=$(find "$OUTPUT_DIR/images/test" -type f 2>/dev/null | wc -l)
    
    # Count annotations with segmentation using Python parser
    if [ -f "$OUTPUT_DIR/annotations/instances_train.json" ]; then
        print_colored "blue" "Analyzing training annotations..."
        count_non_empty_segmentations "$OUTPUT_DIR/annotations/instances_train.json"
    fi
    
    if [ -f "$OUTPUT_DIR/annotations/instances_val.json" ]; then
        print_colored "blue" "Analyzing validation annotations..."
        count_non_empty_segmentations "$OUTPUT_DIR/annotations/instances_val.json"
    fi
    
    if [ -f "$OUTPUT_DIR/annotations/instances_test.json" ]; then
        print_colored "blue" "Analyzing test annotations..."
        count_non_empty_segmentations "$OUTPUT_DIR/annotations/instances_test.json"
    fi
    
    # Display statistics
    print_colored "blue" "Dataset statistics:"
    echo "- Training: $TRAIN_IMAGES images"
    echo "- Validation: $VAL_IMAGES images"
    if [ $TEST_IMAGES -gt 0 ]; then
        echo "- Testing: $TEST_IMAGES images"
    else
        echo "- Testing: No test set found"
    fi
    
    print_colored "blue" "COCO dataset with $SEG_TYPE segmentation masks is ready to use!"
    print_colored "yellow" "Dataset path: $OUTPUT_DIR"
    echo "- Images: $OUTPUT_DIR/images/{train,val,test}"
    echo "- Annotations: $OUTPUT_DIR/annotations/instances_{train,val,test}.json"
    
    # Create a dataset.yaml file for YOLOv8
    YAML_FILE="$OUTPUT_DIR/dataset.yaml"
    print_colored "blue" "Creating YOLOv8 compatible dataset.yaml file..."
    cat > "$YAML_FILE" << EOF
# COCO dataset with $SEG_TYPE segmentation, converted from VOC
path: $OUTPUT_DIR
train: images/train
val: images/val
test: images/test

# Classes
nc: 20
names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
       'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
EOF
    
    print_colored "green" "Created dataset.yaml at $YAML_FILE"
    print_colored "yellow" "To train YOLOv8 segmentation model, run:"
    echo "yolo segment train data=$YAML_FILE model=yolov8n-seg.pt epochs=50"
else
    print_colored "red" "Error during conversion. Check the error messages above."
    exit 1
fi 