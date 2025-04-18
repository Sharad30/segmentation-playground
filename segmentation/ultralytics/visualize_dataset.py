#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import random
import yaml

def read_yolo_instances(label_path):
    """
    Read instance annotations from a YOLO label file
    
    Args:
        label_path: Path to label file
        
    Returns:
        List of instances with class_id, bbox, and polygon points
    """
    instances = []
    
    # Check if label file exists
    if not Path(label_path).exists():
        return instances
        
    # Read label file
    with open(label_path, "r") as f:
        for line in f.readlines():
            # Skip empty lines
            if not line.strip():
                continue
                
            # Parse line
            values = line.strip().split()
            
            # Extract class ID and bounding box coordinates
            cls_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:5])
            
            # Extract polygon points
            points = list(map(float, values[5:]))
            
            # Create instance
            instance = {
                "class_id": cls_id,
                "bbox": [x_center, y_center, width, height],
                "points": points
            }
            
            instances.append(instance)
    
    return instances

def get_class_names(dataset_yaml):
    """
    Get class names from YAML file
    
    Args:
        dataset_yaml: Path to dataset YAML file
        
    Returns:
        Dictionary mapping class IDs to names
    """
    yaml_path = Path(dataset_yaml)
    
    if not yaml_path.exists():
        return {}
    
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        # Get names field from YAML
        if "names" in data:
            # Handle list format
            if isinstance(data["names"], list):
                return {i: name for i, name in enumerate(data["names"])}
            # Handle dict format
            elif isinstance(data["names"], dict):
                return {int(k): v for k, v in data["names"].items()}
    except Exception as e:
        print(f"Error loading class names from {yaml_path}: {e}")
    
    return {}

def generate_class_colors(num_classes, seed=42):
    """
    Generate distinct colors for each class
    
    Args:
        num_classes: Number of classes
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping class IDs to colors
    """
    np.random.seed(seed)
    
    # Generate distinct colors
    colors = {}
    for i in range(num_classes):
        h = i / num_classes
        s = 0.8
        v = 0.9
        
        # Convert HSV to RGB
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        
        if h < 1/6:
            r, g, b = c, x, 0
        elif h < 2/6:
            r, g, b = x, c, 0
        elif h < 3/6:
            r, g, b = 0, c, x
        elif h < 4/6:
            r, g, b = 0, x, c
        elif h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        # Add offset
        r, g, b = r + m, g + m, b + m
        
        colors[i] = (r, g, b)
    
    return colors

def visualize_multiclass(dataset_path, image_name, dataset_yaml=None, output_dir="visualizations", 
                         alpha=0.5, show_labels=True, show_scores=False):
    """
    Visualize multi-class instance segmentation annotations
    
    Args:
        dataset_path: Path to the YOLO dataset
        image_name: Image filename (with or without extension)
        dataset_yaml: Path to dataset YAML file with class names
        output_dir: Directory to save visualizations
        alpha: Transparency of the mask overlay (0.0 to 1.0)
        show_labels: Whether to show class labels
        show_scores: Whether to show confidence scores (if available)
    """
    dataset_path = Path(dataset_path)
    
    # Get class names
    class_names = {}
    if dataset_yaml:
        class_names = get_class_names(dataset_yaml)
    
    # If class names not loaded, use generic names
    if not class_names:
        print("Using generic class names")
        class_names = {i: f"class_{i}" for i in range(100)}
    
    # Remove extension if present
    image_name = Path(image_name).stem
    
    # Try to find the image in train or val directory
    image_path = None
    label_path = None
    
    for split in ['train', 'val']:
        # Check different image extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = dataset_path / 'images' / split / f"{image_name}{ext}"
            if img_path.exists():
                image_path = img_path
                label_path = dataset_path / 'labels' / split / f"{image_name}.txt"
                break
        if image_path is not None:
            break
    
    if image_path is None:
        print(f"Error: Image '{image_name}' not found in dataset")
        return
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        return
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Read instances
    instances = read_yolo_instances(label_path)
    
    if not instances:
        print(f"Warning: No instances found for image: {image_name}")
    
    # Get unique class IDs
    class_ids = sorted(set(instance["class_id"] for instance in instances))
    num_classes = max(class_ids) + 1 if class_ids else 0
    
    # Generate colors for each class
    colors = generate_class_colors(max(num_classes, 20))
    
    # Create figure for full visualization
    plt.figure(figsize=(12, 10))
    plt.title(f"Image: {image_name} - {len(instances)} instances across {len(class_ids)} classes")
    
    # Show the original image
    plt.imshow(image)
    
    # Create a mask overlay
    mask_overlay = np.zeros_like(image, dtype=np.float32)
    
    # Store instances by class for legend
    instances_by_class = {}
    
    # Draw each instance
    for i, instance in enumerate(instances):
        cls_id = instance["class_id"]
        bbox = instance["bbox"]
        points = instance["points"]
        
        # Track instances by class
        if cls_id not in instances_by_class:
            instances_by_class[cls_id] = 0
        instances_by_class[cls_id] += 1
        
        # Get color for this class
        color = colors.get(cls_id, (random.random(), random.random(), random.random()))
        
        # Convert normalized bbox to pixel coordinates
        x_center, y_center, bbox_width, bbox_height = bbox
        x_min = int((x_center - bbox_width / 2) * width)
        y_min = int((y_center - bbox_height / 2) * height)
        x_max = int((x_center + bbox_width / 2) * width)
        y_max = int((y_center + bbox_height / 2) * height)
        
        # Draw bounding box
        plt.gca().add_patch(plt.Rectangle(
            (x_min, y_min), 
            x_max - x_min, 
            y_max - y_min, 
            fill=False, 
            color=color, 
            linewidth=2
        ))
        
        # Get class name
        class_name = class_names.get(cls_id, f"class_{cls_id}")
        
        # Add label if requested
        if show_labels:
            plt.text(
                x_min, 
                y_min - 5, 
                f"{class_name}",
                color=color,
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round')
            )
        
        # Convert normalized polygon points to pixel coordinates
        if len(points) >= 6:  # At least 3 points (x1,y1,x2,y2,x3,y3)
            polygon_points = []
            for j in range(0, len(points), 2):
                if j + 1 < len(points):
                    x = int(points[j] * width)
                    y = int(points[j + 1] * height)
                    polygon_points.append((x, y))
            
            # Convert to numpy array
            polygon_points = np.array(polygon_points, dtype=np.int32)
            
            # Draw polygon
            plt.gca().add_patch(plt.Polygon(
                polygon_points, 
                fill=True, 
                color=color, 
                alpha=0.3
            ))
            
            # Fill polygon in the mask overlay
            cv2.fillPoly(mask_overlay, [polygon_points], color=(color[0]*255, color[1]*255, color[2]*255))
    
    plt.axis('off')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualization
    output_path = Path(output_dir) / f"{image_name}_multiclass.png"
    plt.savefig(str(output_path), bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Create a blended image with the mask overlay
    mask_overlay = mask_overlay.astype(np.uint8)
    blended = cv2.addWeighted(image, 1.0, mask_overlay, alpha, 0)
    
    # Save blended image
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.axis('off')
    plt.title(f"Mask Overlay ({len(instances)} instances)")
    blend_path = Path(output_dir) / f"{image_name}_mask_overlay.png"
    plt.savefig(str(blend_path), bbox_inches='tight')
    print(f"Mask overlay saved to: {blend_path}")
    
    # Create per-class visualizations
    if len(class_ids) > 1:  # Only if there are multiple classes
        # Create a figure for per-class visualization
        fig, axes = plt.subplots(1, len(class_ids), figsize=(5*len(class_ids), 5))
        
        # Handle case where there's only one class
        if len(class_ids) == 1:
            axes = [axes]
        
        # Process each class
        for i, cls_id in enumerate(class_ids):
            ax = axes[i]
            
            # Get class name
            class_name = class_names.get(cls_id, f"class_{cls_id}")
            
            # Show the original image
            ax.imshow(image)
            ax.set_title(f"{class_name} - {instances_by_class[cls_id]} instances")
            ax.axis('off')
            
            # Get color for this class
            color = colors.get(cls_id, (random.random(), random.random(), random.random()))
            
            # Create mask for this class
            class_mask = np.zeros_like(image, dtype=np.float32)
            
            # Draw instances for this class
            for instance in instances:
                if instance["class_id"] != cls_id:
                    continue
                    
                bbox = instance["bbox"]
                points = instance["points"]
                
                # Convert normalized bbox to pixel coordinates
                x_center, y_center, bbox_width, bbox_height = bbox
                x_min = int((x_center - bbox_width / 2) * width)
                y_min = int((y_center - bbox_height / 2) * height)
                x_max = int((x_center + bbox_width / 2) * width)
                y_max = int((y_center + bbox_height / 2) * height)
                
                # Draw bounding box
                rect = plt.Rectangle(
                    (x_min, y_min), 
                    x_max - x_min, 
                    y_max - y_min, 
                    fill=False, 
                    color=color, 
                    linewidth=2
                )
                ax.add_patch(rect)
                
                # Convert normalized polygon points to pixel coordinates
                if len(points) >= 6:
                    polygon_points = []
                    for j in range(0, len(points), 2):
                        if j + 1 < len(points):
                            x = int(points[j] * width)
                            y = int(points[j + 1] * height)
                            polygon_points.append((x, y))
                    
                    # Convert to numpy array
                    polygon_points = np.array(polygon_points, dtype=np.int32)
                    
                    # Draw polygon
                    poly = plt.Polygon(
                        polygon_points, 
                        fill=True, 
                        color=color, 
                        alpha=0.3
                    )
                    ax.add_patch(poly)
                    
                    # Fill polygon in the class mask
                    cv2.fillPoly(class_mask, [polygon_points], color=(color[0]*255, color[1]*255, color[2]*255))
            
            # Overlay the class mask on the image
            ax.imshow(class_mask / 255, alpha=0.4)
        
        # Save per-class visualizations
        per_class_path = Path(output_dir) / f"{image_name}_per_class.png"
        plt.tight_layout()
        plt.savefig(str(per_class_path), bbox_inches='tight')
        print(f"Per-class visualization saved to: {per_class_path}")
    
    # Print statistics
    print(f"\nFound {len(instances)} instances across {len(class_ids)} classes:")
    class_counts = {}
    for instance in instances:
        cls_id = instance["class_id"]
        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
    
    for cls_id, count in sorted(class_counts.items()):
        class_name = class_names.get(cls_id, f"class_{cls_id}")
        print(f"  {class_name} (ID: {cls_id}): {count} instances")
    
    return instances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize multi-class instance segmentation annotations")
    parser.add_argument("--dataset", type=str, default="datasets/yolo_voc_all_classes", 
                       help="Path to YOLO dataset")
    parser.add_argument("--image", type=str, required=True, 
                       help="Image filename (with or without extension)")
    parser.add_argument("--yaml", type=str, 
                       help="Path to dataset YAML file with class names")
    parser.add_argument("--output", type=str, default="visualizations", 
                       help="Output directory for visualizations")
    parser.add_argument("--alpha", type=float, default=0.5, 
                       help="Transparency of mask overlay (0.0 to 1.0)")
    parser.add_argument("--no_labels", action="store_false", dest="show_labels", 
                       help="Don't show class labels")
    
    args = parser.parse_args()
    
    # If YAML not provided, try to use the one in the dataset
    if not args.yaml and Path(args.dataset).exists():
        potential_yaml = Path(args.dataset) / "dataset.yaml"
        if potential_yaml.exists():
            args.yaml = str(potential_yaml)
            print(f"Using dataset YAML from: {args.yaml}")
    
    visualize_multiclass(
        dataset_path=args.dataset,
        image_name=args.image,
        dataset_yaml=args.yaml,
        output_dir=args.output,
        alpha=args.alpha,
        show_labels=args.show_labels
    ) 