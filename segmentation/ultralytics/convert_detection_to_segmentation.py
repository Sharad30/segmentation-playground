#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import glob
import shutil
import random
import sys

def convert_dataset_to_segmentation(input_dir, output_dir, simplify=True):
    """
    Convert existing dataset to segmentation format
    
    Args:
        input_dir: Path to input dataset with train/val splits
        output_dir: Path to save segmentation format dataset
        simplify: Whether to use simplified polygons
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Debug - let's check what directories we actually have
    print(f"Input directory: {input_dir} (exists: {input_dir.exists()})")
    
    # Print all subdirectories and files
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(input_dir):
        level = root.replace(str(input_dir), '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        
        # Print a sample of files if there are too many
        sample_files = files[:5] if len(files) > 5 else files
        for f in sample_files:
            print(f"{sub_indent}{f}")
        if len(files) > 5:
            print(f"{sub_indent}... and {len(files) - 5} more files")
    
    # Now try a basic approach - look for any image files recursively
    all_image_files = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    for ext in image_extensions:
        # Use glob for recursive search
        all_image_files.extend(glob.glob(f"{input_dir}/**/*{ext}", recursive=True))
    
    print(f"\nFound {len(all_image_files)} total image files recursively")
    
    if len(all_image_files) > 0:
        print(f"Sample images found:")
        for img in all_image_files[:5]:
            print(f"  {img}")
    
    # Try to find which structure we're dealing with
    has_train_val = False
    train_dir = val_dir = None
    
    # Check if we have a nested structure
    if (input_dir / "images" / "train").exists() and (input_dir / "images" / "val").exists():
        has_train_val = True
        train_dir = input_dir / "images" / "train"
        val_dir = input_dir / "images" / "val"
        print(f"\nFound nested train/val structure:")
        print(f"  Train dir: {train_dir}")
        print(f"  Val dir: {val_dir}")
    
    # If we have train/val directories but no images found, there's something wrong
    if has_train_val:
        train_images = []
        val_images = []
        
        for ext in image_extensions:
            train_images.extend(list(train_dir.glob(f"*{ext}")))
            val_images.extend(list(val_dir.glob(f"*{ext}")))
        
        print(f"Train images found: {len(train_images)}")
        print(f"Val images found: {len(val_images)}")
        
        if len(train_images) == 0 and len(val_images) == 0:
            print(f"ERROR: Directories exist but no images found with extensions {image_extensions}")
            print(f"Please check file extensions and permissions")
            
            # List some files in these directories to see what's there
            print(f"\nFiles in train directory:")
            train_files = list(train_dir.glob("*"))
            for f in train_files[:5]:
                print(f"  {f.name}")
            
            print(f"\nFiles in val directory:")
            val_files = list(val_dir.glob("*"))
            for f in val_files[:5]:
                print(f"  {f.name}")
                
            # If we have files but no images, check their extensions
            if train_files or val_files:
                print("\nDetected extensions:")
                all_extensions = set()
                for f in train_files + val_files:
                    if f.is_file():
                        ext = f.suffix.lower()
                        all_extensions.add(ext)
                print(f"  {', '.join(all_extensions)}")
    
    # Try using all image files found anywhere
    if all_image_files:
        print(f"\nUsing {len(all_image_files)} images found recursively")
        
        # Separate images by train/val based on path or random split
        train_images = []
        val_images = []
        
        for img_path in all_image_files:
            img_path = Path(img_path)
            
            # Check if image is already in a train or val directory
            if "train" in str(img_path):
                train_images.append(img_path)
            elif "val" in str(img_path):
                val_images.append(img_path)
            else:
                # If not in train or val directory, add to train by default
                train_images.append(img_path)
        
        # If all images went to train, do a random split
        if val_images == []:
            random.shuffle(train_images)
            split_idx = int(len(train_images) * 0.8)
            val_images = train_images[split_idx:]
            train_images = train_images[:split_idx]
        
        print(f"Final split: {len(train_images)} train images, {len(val_images)} val images")
        
        # Process train images
        for img_path in tqdm(train_images, desc="Processing train images"):
            process_image_path(img_path, input_dir, output_dir, "train", simplify)
        
        # Process val images
        for img_path in tqdm(val_images, desc="Processing val images"):
            process_image_path(img_path, input_dir, output_dir, "val", simplify)
    else:
        print("ERROR: No image files found in any directory")
        print("Please check that the dataset directory contains image files")
        return
    
    # Create dataset.yaml file
    yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val

nc: 20
names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
       'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
"""
    
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print(f"\nConversion complete! Dataset saved to {output_dir}")
    print(f"YAML configuration file created at {output_dir / 'dataset.yaml'}")
    print(f"\nTo train your model, run:")
    print(f"python segmentation/ultralytics/train.py --data {output_dir.absolute()}/dataset.yaml --epochs 20 --batch 16 --imgsz 640 --device 0")

def process_image_path(img_path, input_dir, output_dir, split, simplify):
    """Process a single image path, finding the label and converting to segmentation format"""
    img_path = Path(img_path)
    img_name = img_path.name
    img_stem = img_path.stem
    
    # Find the label file by looking in various possible locations
    label_path = None
    potential_label_paths = []
    
    # Try to find 'labels' directory at the same level as the image directory
    img_dir = img_path.parent
    parent_dir = img_dir.parent
    
    # Check if image is in 'images' directory and look for matching 'labels' directory
    if img_dir.name == "images" or img_dir.name == "train" or img_dir.name == "val":
        label_dir = parent_dir / "labels"
        if label_dir.exists():
            potential_label_paths.append(label_dir / f"{img_stem}.txt")
    
    # Look for labels in same directory as image
    potential_label_paths.append(img_dir / f"{img_stem}.txt")
    
    # Look for labels in a 'labels' subdirectory alongside the images
    potential_label_paths.append(parent_dir / "labels" / f"{img_stem}.txt")
    
    # For nested train/val structure
    if "train" in str(img_dir) or "val" in str(img_dir):
        split_name = "train" if "train" in str(img_dir) else "val"
        
        # Look for labels in corresponding labels directory
        potential_label_paths.append(parent_dir.parent / "labels" / split_name / f"{img_stem}.txt")
    
    # Check for different extensions or naming patterns
    for base_path in list(potential_label_paths):
        potential_label_paths.append(Path(str(base_path).replace(".jpg", "")))
        potential_label_paths.append(Path(str(base_path).replace(".png", "")))
        potential_label_paths.append(Path(f"{base_path.parent}/{img_stem}_jpg.txt"))
        potential_label_paths.append(Path(f"{base_path.parent}/{img_stem}.jpg.txt"))
    
    # Check if any potential path exists
    for path in potential_label_paths:
        if path.exists():
            label_path = path
            break
    
    if not label_path:
        # Try a recursive search for the label file
        label_files = list(input_dir.glob(f"**/{img_stem}.txt")) + list(input_dir.glob(f"**/{img_stem}_jpg.txt"))
        if label_files:
            label_path = label_files[0]
    
    if not label_path:
        print(f"Warning: No label found for {img_name}, skipping")
        return
    
    # Copy image to output directory
    shutil.copy(img_path, output_dir / "images" / split / img_name)
    
    # Convert detection label to segmentation format
    convert_label_to_segmentation(img_path, label_path, output_dir / "labels" / split / f"{img_stem}.txt", simplify)

def convert_label_to_segmentation(img_path, label_path, output_path, simplify):
    """Convert detection label to segmentation format with polygon points"""
    try:
        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return
        
        h, w = img.shape[:2]
        
        # Create new label file
        with open(output_path, "w") as outfile:
            # Read original label file
            with open(label_path, "r") as infile:
                for line in infile:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    # Extract bounding box in YOLO format
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Keep the bounding box part the same
                    outfile.write(f"{cls_id} {x_center} {y_center} {width} {height}")
                    
                    # If this is already a segmentation label with polygon points, copy them
                    if len(parts) > 5:
                        # This already has polygon points, just copy them
                        for i in range(5, len(parts)):
                            outfile.write(f" {parts[i]}")
                        outfile.write("\n")
                        continue
                    
                    # Otherwise, generate polygon points from the bounding box
                    # Convert normalized coordinates to pixel coordinates
                    x_center_px = x_center * w
                    y_center_px = y_center * h
                    width_px = width * w
                    height_px = height * h
                    
                    x1 = max(0, int(x_center_px - width_px / 2))
                    y1 = max(0, int(y_center_px - height_px / 2))
                    x2 = min(w - 1, int(x_center_px + width_px / 2))
                    y2 = min(h - 1, int(y_center_px + height_px / 2))
                    
                    # Create polygon points
                    polygon = []
                    
                    # Add corners
                    polygon.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                    
                    # Add midpoints
                    polygon.extend([(x1 + (x2-x1)//2, y1), (x2, y1 + (y2-y1)//2),
                                   (x1 + (x2-x1)//2, y2), (x1, y1 + (y2-y1)//2)])
                    
                    # Add more points if desired
                    if not simplify:
                        polygon.extend([(x1 + (x2-x1)//3, y1), (x1 + 2*(x2-x1)//3, y1),
                                       (x2, y1 + (y2-y1)//3), (x2, y1 + 2*(y2-y1)//3),
                                       (x1 + (x2-x1)//3, y2), (x1 + 2*(x2-x1)//3, y2),
                                       (x1, y1 + (y2-y1)//3), (x1, y1 + 2*(y2-y1)//3)])
                    
                    # Write polygon points
                    for px, py in polygon:
                        outfile.write(f" {px/w:.6f} {py/h:.6f}")
                    
                    outfile.write("\n")
    except Exception as e:
        print(f"Error processing {label_path}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to segmentation format")
    parser.add_argument("--input_dir", type=str, 
                        default="/home/ubuntu/sharad/segmentation-playground/datasets/voc_yolo_detection",
                        help="Path to input dataset directory")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/ubuntu/sharad/segmentation-playground/datasets/voc_yolo_segmentation",
                        help="Path to save segmentation format dataset")
    parser.add_argument("--simplify", action="store_true", help="Use simplified polygons")
    
    args = parser.parse_args()
    convert_dataset_to_segmentation(args.input_dir, args.output_dir, args.simplify)