#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm
import multiprocessing
from functools import partial
from skimage import measure
from skimage.measure import find_contours, approximate_polygon

# VOC classes mapping
VOC_CLASSES = {
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
    6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
    11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
    16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor',
    0: 'background', 255: 'void/ignore'
}

# Standard mapping from VOC class ID to YOLO class ID (0-19)
VOC_TO_YOLO_MAPPING = {i: idx for idx, i in enumerate(range(1, 21))}

def mask_to_polygons(mask, epsilon=1.0, min_area=20):
    """
    Convert a binary mask to polygons
    
    Args:
        mask: Binary mask
        epsilon: Polygon approximation error tolerance
        min_area: Minimum area for a valid polygon
        
    Returns:
        List of instances with bounding boxes and polygons
    """
    instances = []
    
    # Find connected components
    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask)
    
    height, width = mask.shape
    
    for region in regions:
        # Skip small regions
        if region.area < min_area:
            continue
        
        # Get region mask
        region_mask = np.zeros_like(mask, dtype=bool)
        region_mask[labeled_mask == region.label] = True
        
        # Find contours
        contours = find_contours(region_mask.astype(np.uint8), 0.5)
        
        if not contours:
            continue
            
        # Get the largest contour (by number of points)
        contour = max(contours, key=lambda x: len(x))
        
        # Skip contours with too few points
        if len(contour) < 4:
            continue
        
        # Approximate polygon
        polygon = approximate_polygon(contour, epsilon)
        
        # Skip polygons with too few points
        if len(polygon) < 4:
            continue
        
        # Ensure polygon is closed (last point = first point)
        if not np.array_equal(polygon[0], polygon[-1]):
            polygon = np.vstack([polygon, polygon[0]])
        
        # Get bounding box
        min_row, min_col = np.min(polygon, axis=0)
        max_row, max_col = np.max(polygon, axis=0)
        
        # Convert to YOLO format (x_center, y_center, width, height)
        x_center = (min_col + max_col) / 2 / width
        y_center = (min_row + max_row) / 2 / height
        box_width = (max_col - min_col) / width
        box_height = (max_row - min_row) / height
        
        # Skip invalid boxes
        if box_width <= 0 or box_height <= 0:
            continue
        
        # Create flattened polygon points (normalized coordinates)
        polygon_points = []
        for point in polygon:
            # Switch from (row, col) to (x, y)
            y, x = point
            # Normalize coordinates
            x_norm = x / width
            y_norm = y / height
            polygon_points.extend([x_norm, y_norm])
        
        # Create instance
        instance = {
            "bbox": [x_center, y_center, box_width, box_height],
            "polygon": polygon_points
        }
        
        instances.append(instance)
    
    return instances

def process_image(image_id, voc_path, output_path, split, epsilon=1.0, min_area=20, verbose=False):
    """
    Process a single image, find all class instances, and save as YOLO format
    
    Args:
        image_id: Image ID
        voc_path: Path to VOC dataset
        output_path: Path to output directory
        split: Dataset split (train/val)
        epsilon: Polygon approximation error tolerance
        min_area: Minimum area for a valid polygon
        verbose: Print verbose information
        
    Returns:
        Dictionary with image_id and processed instance counts per class
    """
    voc_path = Path(voc_path)
    output_path = Path(output_path)
    
    # Get paths
    image_file = None
    for ext in ['.jpg', '.jpeg', '.png']:
        img_path = voc_path / "JPEGImages" / f"{image_id}{ext}"
        if img_path.exists():
            image_file = img_path
            break
    
    # Skip if image not found
    if image_file is None:
        if verbose:
            print(f"Image not found for ID: {image_id}")
        return {"image_id": image_id, "instances": {}}
    
    mask_file = voc_path / "SegmentationClass" / f"{image_id}.png"
    
    # Skip if mask not found
    if not mask_file.exists():
        if verbose:
            print(f"Mask not found for ID: {image_id}")
        return {"image_id": image_id, "instances": {}}
    
    try:
        # Read image and mask
        image = Image.open(image_file)
        mask = np.array(Image.open(mask_file))
        
        # Create directories
        (output_path / "images" / split).mkdir(exist_ok=True, parents=True)
        (output_path / "labels" / split).mkdir(exist_ok=True, parents=True)
        
        # Copy image
        image_output = output_path / "images" / split / f"{image_id}.jpg"
        if image_file.suffix.lower() in ['.jpg', '.jpeg']:
            shutil.copy2(image_file, image_output)
        else:
            # Convert to JPEG
            image.save(image_output, "JPEG", quality=95)
        
        # Create label file
        label_output = output_path / "labels" / split / f"{image_id}.txt"
        
        # Count instances per class
        instance_counts = {}
        
        # Process each class (1-20)
        with open(label_output, "w") as f:
            for voc_class_id in range(1, 21):
                # Get YOLO class ID (0-19)
                yolo_class_id = VOC_TO_YOLO_MAPPING[voc_class_id]
                
                # Create binary mask for this class
                binary_mask = (mask == voc_class_id).astype(np.uint8)
                
                # Skip if no pixels for this class
                if np.sum(binary_mask) == 0:
                    continue
                
                # Convert mask to instances (polygons)
                instances = mask_to_polygons(binary_mask, epsilon=epsilon, min_area=min_area)
                
                # Skip if no instances found
                if not instances:
                    continue
                
                # Update instance count for this class
                instance_counts[yolo_class_id] = len(instances)
                
                # Write instances to label file
                for instance in instances:
                    bbox = instance["bbox"]
                    polygon = instance["polygon"]
                    
                    # Write YOLO format line: class_id x_center y_center width height polygon_points...
                    line = f"{yolo_class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                    for point in polygon:
                        line += f" {point:.6f}"
                    f.write(line + "\n")
        
        return {"image_id": image_id, "instances": instance_counts}
    
    except Exception as e:
        if verbose:
            print(f"Error processing {image_id}: {e}")
        return {"image_id": image_id, "instances": {}}

def convert_voc_to_yolo_multiclass(voc_path, output_path, epsilon=1.0, min_area=20, workers=1, verbose=True):
    """
    Convert VOC dataset to YOLO format for instance segmentation with all classes
    
    Args:
        voc_path: Path to VOC dataset
        output_path: Path to output directory
        epsilon: Polygon approximation error tolerance
        min_area: Minimum area for a valid polygon
        workers: Number of worker processes
        verbose: Print verbose information
    """
    voc_path = Path(voc_path)
    output_path = Path(output_path)
    
    # Verify VOC dataset structure
    required_dirs = ["JPEGImages", "SegmentationClass", "ImageSets/Segmentation"]
    missing_dirs = [d for d in required_dirs if not (voc_path / d).exists()]
    if missing_dirs:
        print(f"Error: Missing required directories: {missing_dirs}")
        return
    
    # Create output directories
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Read train/val splits
    train_ids = []
    val_ids = []
    
    train_file = voc_path / "ImageSets" / "Segmentation" / "train.txt"
    val_file = voc_path / "ImageSets" / "Segmentation" / "val.txt"
    
    if train_file.exists():
        with open(train_file, "r") as f:
            train_ids = [line.strip() for line in f if line.strip()]
    
    if val_file.exists():
        with open(val_file, "r") as f:
            val_ids = [line.strip() for line in f if line.strip()]
    
    if not train_ids and not val_ids:
        print("Error: No images found in train or val splits")
        return
    
    if verbose:
        print(f"Found {len(train_ids)} train images and {len(val_ids)} val images")
    
    # Create YAML file
    yaml_data = {
        "path": str(output_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "",
        "nc": 20,
        "names": [VOC_CLASSES[i] for i in range(1, 21)]
    }
    
    with open(output_path / "dataset.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    if verbose:
        print(f"Created dataset.yaml with {yaml_data['nc']} classes")
    
    # Process images
    results = []
    
    # Process training images
    if verbose:
        print("Processing training images...")
    
    if workers > 1:
        # Use multiprocessing
        process_func = partial(
            process_image,
            voc_path=voc_path,
            output_path=output_path,
            split="train",
            epsilon=epsilon,
            min_area=min_area,
            verbose=False
        )
        
        with multiprocessing.Pool(workers) as pool:
            train_results = list(tqdm(pool.imap(process_func, train_ids), total=len(train_ids), disable=not verbose))
        
        results.extend(train_results)
    else:
        # Process sequentially
        for image_id in tqdm(train_ids, disable=not verbose):
            result = process_image(
                image_id=image_id,
                voc_path=voc_path,
                output_path=output_path,
                split="train",
                epsilon=epsilon,
                min_area=min_area,
                verbose=verbose
            )
            results.append(result)
    
    # Process validation images
    if verbose:
        print("Processing validation images...")
    
    if workers > 1:
        # Use multiprocessing
        process_func = partial(
            process_image,
            voc_path=voc_path,
            output_path=output_path,
            split="val",
            epsilon=epsilon,
            min_area=min_area,
            verbose=False
        )
        
        with multiprocessing.Pool(workers) as pool:
            val_results = list(tqdm(pool.imap(process_func, val_ids), total=len(val_ids), disable=not verbose))
        
        results.extend(val_results)
    else:
        # Process sequentially
        for image_id in tqdm(val_ids, disable=not verbose):
            result = process_image(
                image_id=image_id,
                voc_path=voc_path,
                output_path=output_path,
                split="val",
                epsilon=epsilon,
                min_area=min_area,
                verbose=verbose
            )
            results.append(result)
    
    # Count instances
    total_processed = len(results)
    train_images = sum(1 for r in results if r["image_id"] in train_ids and r["instances"])
    val_images = sum(1 for r in results if r["image_id"] in val_ids and r["instances"])
    
    # Count instances per class
    class_instances = {i: 0 for i in range(20)}
    
    for result in results:
        instances = result["instances"]
        for cls_id, count in instances.items():
            class_instances[cls_id] += count
    
    total_instances = sum(class_instances.values())
    
    if verbose:
        print("\nConversion complete!")
        print(f"Total images processed: {total_processed}")
        print(f"  - Train images: {train_images}")
        print(f"  - Val images: {val_images}")
        print(f"Total instances: {total_instances}")
        print("Instances per class:")
        
        for cls_id, count in class_instances.items():
            if count > 0:
                cls_name = VOC_CLASSES[cls_id + 1]  # Convert YOLO ID back to VOC ID for name lookup
                print(f"  - {cls_name} (YOLO ID: {cls_id}): {count} instances")
    
    # Write mapping file for reference
    mapping_file = output_path / "class_mapping.txt"
    with open(mapping_file, "w") as f:
        f.write("VOC Class ID -> YOLO Class ID (Class Name)\n")
        for voc_id, yolo_id in VOC_TO_YOLO_MAPPING.items():
            f.write(f"{voc_id} -> {yolo_id} ({VOC_CLASSES[voc_id]})\n")
    
    if verbose:
        print(f"\nClass mapping saved to {mapping_file}")
        print(f"Dataset configuration saved to {output_path / 'dataset.yaml'}")
    
    return {
        "total_processed": total_processed,
        "train_images": train_images,
        "val_images": val_images,
        "total_instances": total_instances,
        "class_instances": class_instances
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PASCAL VOC dataset to YOLO format for instance segmentation with all classes")
    parser.add_argument("--voc_path", type=str, required=True, help="Path to VOC dataset")
    parser.add_argument("--output_path", type=str, default="datasets/yolo_voc_all_classes", help="Path to output directory")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Polygon approximation error tolerance")
    parser.add_argument("--min_area", type=int, default=20, help="Minimum area for a valid polygon")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (0=auto)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose information")
    
    args = parser.parse_args()
    
    # Set workers to CPU count if 0
    if args.workers == 0:
        args.workers = multiprocessing.cpu_count()
    
    convert_voc_to_yolo_multiclass(
        voc_path=args.voc_path,
        output_path=args.output_path,
        epsilon=args.epsilon,
        min_area=args.min_area,
        workers=args.workers,
        verbose=args.verbose or True
    ) 