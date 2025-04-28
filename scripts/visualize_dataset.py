#!/usr/bin/env python3
import os
import random
import json
import yaml
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import cv2

# Path constants
DATASETS_ROOT = "/home/ubuntu/sharad/segmentation-playground/datasets/processed"

def detect_dataset_format(dataset_path):
    """
    Detect if a dataset is in YOLO or COCO format based on its folder structure.
    
    Returns:
        str: "yolo", "coco", or "unknown"
    """
    dataset_path = Path(dataset_path)
    
    # Check for COCO format first (more specific check)
    if (dataset_path / "annotations").exists() and (dataset_path / "images").exists():
        annotations_dir = dataset_path / "annotations"
        json_files = list(annotations_dir.glob("*.json"))
        
        if json_files:
            print(f"Detected COCO format dataset at {dataset_path} with JSON files: {json_files}")
            try:
                with open(json_files[0], 'r') as f:
                    json_data = json.load(f)
                
                # Check if it has expected COCO structure
                if all(key in json_data for key in ["images", "annotations", "categories"]):
                    print(f"Confirmed COCO format - found required keys in JSON")
                    return "coco"
            except Exception as e:
                print(f"Error reading JSON file: {e}")
    
    # Check for YOLO format
    yaml_files = []
    if (dataset_path / "dataset.yaml").exists():
        yaml_files.append(dataset_path / "dataset.yaml")
    elif (dataset_path / "data.yaml").exists():
        yaml_files.append(dataset_path / "data.yaml")
    
    if yaml_files:
        print(f"Found potential YOLO dataset at {dataset_path} with YAML files: {yaml_files}")
        try:
            with open(yaml_files[0], 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # Check if it has expected YOLO structure
            if all(key in yaml_data for key in ["path", "train", "val"]):
                # Additional check: verify labels directory exists
                if (dataset_path / "labels").exists():
                    print(f"Confirmed YOLO format - found required keys in YAML and labels directory")
                    return "yolo"
                print(f"Missing labels directory for YOLO dataset")
        except Exception as e:
            print(f"Error reading YAML file: {e}")
    
    print(f"Could not determine dataset format for {dataset_path}")
    return "unknown"

def get_available_datasets():
    """Return list of available datasets and their formats."""
    datasets = []
    
    if not os.path.exists(DATASETS_ROOT):
        return []
    
    for item in os.listdir(DATASETS_ROOT):
        dataset_path = os.path.join(DATASETS_ROOT, item)
        if os.path.isdir(dataset_path):
            format_type = detect_dataset_format(dataset_path)
            if format_type != "unknown":
                datasets.append({
                    "name": item,
                    "path": dataset_path,
                    "format": format_type
                })
    
    return datasets

def load_yolo_dataset(dataset_path):
    """Load a YOLO format dataset."""
    dataset_path = Path(dataset_path)
    
    # Find and load the YAML file
    if (dataset_path / "dataset.yaml").exists():
        yaml_file = dataset_path / "dataset.yaml"
    else:
        yaml_file = dataset_path / "data.yaml"
    
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get paths
    base_path = Path(data.get("path", dataset_path))
    if not base_path.exists():
        base_path = dataset_path  # Fallback if path doesn't exist
    
    train_path = base_path / data.get("train", "images/train")
    val_path = base_path / data.get("val", "images/val")
    
    # Get class names
    class_names = data.get("names", [])
    
    return {
        "train_path": train_path,
        "val_path": val_path,
        "class_names": class_names,
        "yaml_data": data
    }

def load_coco_dataset(dataset_path):
    """Load a COCO format dataset."""
    dataset_path = Path(dataset_path)
    
    # Find annotation files
    annotations_path = dataset_path / "annotations"
    train_json = annotations_path / "instances_train.json"
    val_json = annotations_path / "instances_val.json"
    
    # Load annotation files
    train_data = None
    val_data = None
    
    if train_json.exists():
        with open(train_json, 'r') as f:
            train_data = json.load(f)
    
    if val_json.exists():
        with open(val_json, 'r') as f:
            val_data = json.load(f)
    
    # Get class names
    class_names = []
    if train_data and "categories" in train_data:
        categories = sorted(train_data["categories"], key=lambda x: x["id"])
        class_names = [cat["name"] for cat in categories]
    elif val_data and "categories" in val_data:
        categories = sorted(val_data["categories"], key=lambda x: x["id"])
        class_names = [cat["name"] for cat in categories]
    
    return {
        "train_data": train_data,
        "val_data": val_data,
        "class_names": class_names,
        "train_images_path": dataset_path / "images/train",
        "val_images_path": dataset_path / "images/val"
    }

def visualize_yolo_instance(image_path, label_path, class_names):
    """Visualize instance masks for YOLO format image."""
    # Load image
    image = np.array(Image.open(image_path))
    image_filename = os.path.basename(image_path)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    fig.suptitle(f"Image: {image_filename}", fontsize=14)
    ax.imshow(image)
    
    # Check if label file exists
    h, w = image.shape[:2]
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                cls_id = int(parts[0])
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                
                # YOLO segmentation format: class_id x1 y1 x2 y2 ... xn yn
                if len(parts) > 5:  # Segmentation
                    # Extract polygon points
                    polygon = []
                    for i in range(1, len(parts), 2):
                        if i+1 < len(parts):
                            x = float(parts[i]) * w
                            y = float(parts[i+1]) * h
                            polygon.append((x, y))
                    
                    if polygon:
                        # Convert to numpy array for plotting
                        polygon_np = np.array(polygon)
                        
                        # Generate a random color for the mask
                        color = np.random.rand(3,)
                        
                        # Draw filled polygon with higher alpha value for better visibility
                        ax.fill(polygon_np[:, 0], polygon_np[:, 1], color=color, alpha=0.7)
                        ax.plot(polygon_np[:, 0], polygon_np[:, 1], color=color, linewidth=2)
                        
                        # Add label at centroid
                        centroid_x = np.mean(polygon_np[:, 0])
                        centroid_y = np.mean(polygon_np[:, 1])
                        ax.text(centroid_x, centroid_y, cls_name, fontsize=12, 
                                color='white', bbox=dict(facecolor=color, alpha=0.8))
                
    ax.set_title("Instance Segmentation")
    ax.axis('off')
    
    # Adjust layout to make room for the title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

def visualize_yolo_bbox(image_path, label_path, class_names):
    """Visualize bounding boxes for YOLO format image."""
    # Load image
    image = np.array(Image.open(image_path))
    image_filename = os.path.basename(image_path)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    fig.suptitle(f"Image: {image_filename}", fontsize=14)
    ax.imshow(image)
    
    # Check if label file exists
    h, w = image.shape[:2]
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                cls_id = int(parts[0])
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                
                # YOLO bbox format: class_id x_center y_center width height
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                width = float(parts[3]) * w
                height = float(parts[4]) * h
                
                # Calculate corners from center
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                
                # Generate a random color
                color = np.random.rand(3,)
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (x_min, y_min), width, height, 
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label at top-left corner
                ax.text(x_min, y_min - 5, cls_name, fontsize=12,
                        color='white', bbox=dict(facecolor=color, alpha=0.8))
    
    ax.set_title("Bounding Boxes")
    ax.axis('off')
    
    # Adjust layout to make room for the title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

def visualize_coco_instance(image_path, annotations, image_id, class_names):
    """Visualize instance masks for COCO format image."""
    try:
        # Load image
        image = np.array(Image.open(image_path))
        image_filename = os.path.basename(image_path)
        
        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 8))
        fig.suptitle(f"Image: {image_filename}", fontsize=14)
        ax.imshow(image)
        
        # Find annotations for this image
        image_annotations = [ann for ann in annotations if ann["image_id"] == image_id]
        print(f"Found {len(image_annotations)} annotations for image ID {image_id}")
        
        # Track if we're showing any true segmentation masks
        has_true_segmentation = False
        has_rectangular_segmentation = False
        
        # Draw each annotation
        for ann in image_annotations:
            category_id = ann["category_id"]
            cls_name = class_names[category_id-1] if 0 <= category_id-1 < len(class_names) else f"Class {category_id}"
            
            # Generate a random color
            color = np.random.rand(3,)
            
            # Check for segmentation data
            if "segmentation" in ann and ann["segmentation"]:
                for i, segmentation in enumerate(ann["segmentation"]):
                    # COCO segmentation is a flat list of [x1, y1, x2, y2, ...]
                    if len(segmentation) >= 6:  # At least 3 points
                        # Convert flat list to array of points
                        points = np.array(segmentation).reshape(-1, 2)
                        
                        # Check if this is just a rectangle (likely converted from a bbox)
                        is_rectangle = False
                        if len(points) == 4:  # 4 points could be a rectangle
                            # Check if it forms a rectangle
                            x_coords = points[:, 0]
                            y_coords = points[:, 1]
                            if len(set(x_coords)) == 2 and len(set(y_coords)) == 2:
                                is_rectangle = True
                                has_rectangular_segmentation = True
                                print(f"Detected rectangular segmentation for {cls_name}")
                        
                        # Draw filled polygon with higher alpha value for better visibility
                        ax.fill(points[:, 0], points[:, 1], color=color, alpha=0.7)
                        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2)
                        
                        # Add label at centroid with different text depending on whether it's a real mask
                        centroid_x = np.mean(points[:, 0])
                        centroid_y = np.mean(points[:, 1])
                        
                        label_text = cls_name
                        if is_rectangle:
                            label_text = f"{cls_name}\n(box mask)"
                        else:
                            has_true_segmentation = True
                            
                        ax.text(centroid_x, centroid_y, label_text, fontsize=12, 
                                color='white', bbox=dict(facecolor=color, alpha=0.8))
                    else:
                        print(f"  Warning: Segment has only {len(segmentation)//2} points, skipping")
            elif "bbox" in ann:
                # If there's no segmentation but there is a bbox, use the bbox with a clear indicator
                print(f"No segmentation data for annotation with class {cls_name}, using bbox")
                bbox = ann["bbox"]
                # COCO bbox format: [x, y, width, height]
                x, y, width, height = bbox
                
                # Draw rectangle with visible alpha to show it's a fallback
                rect = patches.Rectangle(
                    (x, y), width, height, 
                    linewidth=2, edgecolor=color, facecolor=color, alpha=0.5
                )
                ax.add_patch(rect)
                
                # Add label with clarification
                ax.text(x + width/2, y + height/2, f"{cls_name}\n(bbox only)", 
                        fontsize=10, color='white', ha='center', va='center',
                        bbox=dict(facecolor=color, alpha=0.8))
        
        # Add warning text if we're only showing rectangular "segmentations"
        if has_rectangular_segmentation and not has_true_segmentation:
            fig.text(0.5, 0.01, 
                    "⚠️ Warning: This image only contains rectangular segmentation masks (converted from bounding boxes)",
                    ha='center', va='bottom', color='red', fontsize=12,
                    bbox=dict(facecolor='yellow', alpha=0.3))
        
        ax.set_title("Instance Segmentation")
        ax.axis('off')
        
        # Adjust layout to make room for the title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig
    except Exception as e:
        st.error(f"Error visualizing COCO instance: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        # Return a blank figure in case of error
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        ax.axis('off')
        return fig

def visualize_coco_bbox(image_path, annotations, image_id, class_names):
    """Visualize bounding boxes for COCO format image."""
    try:
        # Load image
        image = np.array(Image.open(image_path))
        image_filename = os.path.basename(image_path)
        
        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 8))
        fig.suptitle(f"Image: {image_filename}", fontsize=14)
        ax.imshow(image)
        
        # Find annotations for this image
        image_annotations = [ann for ann in annotations if ann["image_id"] == image_id]
        
        # Draw each annotation
        for ann in image_annotations:
            category_id = ann["category_id"]
            cls_name = class_names[category_id-1] if 0 <= category_id-1 < len(class_names) else f"Class {category_id}"
            
            # Generate a random color
            color = np.random.rand(3,)
            
            # Draw bounding box
            if "bbox" in ann:
                bbox = ann["bbox"]
                # COCO bbox format: [x, y, width, height]
                x, y, width, height = bbox
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (x, y), width, height, 
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label at top-left corner
                ax.text(x, y - 5, cls_name, fontsize=12,
                        color='white', bbox=dict(facecolor=color, alpha=0.8))
        
        ax.set_title("Bounding Boxes")
        ax.axis('off')
        
        # Adjust layout to make room for the title
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig
    except Exception as e:
        st.error(f"Error visualizing COCO bounding boxes: {str(e)}")
        # Return a blank figure in case of error
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        ax.axis('off')
        return fig

def main():
    st.set_page_config(page_title="Dataset Visualizer", layout="wide")
    
    st.title("Dataset Visualizer")
    st.markdown("Explore datasets in the processed folder with visualization options.")
    
    # Get available datasets
    datasets = get_available_datasets()
    
    if not datasets:
        st.error("No datasets found in the processed folder.")
        return
    
    # Sidebar for dataset selection
    st.sidebar.header("Dataset Selection")
    
    # Create dropdown for dataset selection
    dataset_names = [f"{d['name']} ({d['format'].upper()})" for d in datasets]
    selected_dataset_idx = st.sidebar.selectbox("Select Dataset", range(len(datasets)), format_func=lambda x: dataset_names[x])
    
    selected_dataset = datasets[selected_dataset_idx]
    dataset_format = selected_dataset["format"]
    dataset_path = selected_dataset["path"]
    
    st.sidebar.markdown(f"**Format:** {dataset_format.upper()}")
    st.sidebar.markdown(f"**Path:** {dataset_path}")
    
    # Log dataset info for debugging
    print(f"Selected dataset: {selected_dataset['name']}")
    print(f"Dataset format: {dataset_format}")
    print(f"Dataset path: {dataset_path}")
    
    # Choose dataset split
    split = st.sidebar.radio("Select Split", ["train", "val"])
    
    # Choose visualization type
    st.sidebar.header("Visualization Options")
    
    if dataset_format == "yolo":
        vis_options = ["Bounding Boxes", "Instance Segmentation"]
        default_index = 0
    else:  # COCO format
        vis_options = ["Bounding Boxes", "Instance Segmentation"]
        default_index = 1
    
    vis_type = st.sidebar.radio(
        "Select Visualization Type",
        vis_options,
        index=default_index
    )
    
    # Process based on dataset format
    try:
        print(f"Visualizing dataset with format {dataset_format}, split {split}, visualization type {vis_type}")
        
        if "coco" in dataset_format.lower():
            print("Using COCO visualization path")
            visualize_coco_dataset(dataset_path, split, vis_type)
        elif "yolo" in dataset_format.lower():
            print("Using YOLO visualization path")
            visualize_yolo_dataset(dataset_path, split, vis_type)
        else:
            st.error(f"Unsupported dataset format: {dataset_format}")
    except Exception as e:
        import traceback
        st.error(f"Error visualizing dataset: {str(e)}")
        st.code(traceback.format_exc())
        st.info("Please try selecting a different dataset or visualization type")

def visualize_yolo_dataset(dataset_path, split, vis_type):
    """Handle visualization for YOLO format datasets"""
    # Load dataset
    dataset_info = load_yolo_dataset(dataset_path)
    class_names = dataset_info["class_names"]
    
    # Get image path based on split
    if split == "train":
        images_path = dataset_info["train_path"]
    else:
        images_path = dataset_info["val_path"]
    
    # Get list of images
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.error(f"No images found in {images_path}")
        return
    
    # Randomly pick an image or allow user to select
    selection_method = st.sidebar.radio("Image Selection", ["Random", "Choose"])
    
    if selection_method == "Random":
        selected_img = random.choice(image_files)
        st.sidebar.button("New Random Image", key="random")
    else:
        selected_img = st.sidebar.selectbox("Select Image", image_files)
    
    # YOLO format visualizations
    image_path = os.path.join(images_path, selected_img)
    
    # Get corresponding label file
    label_file = os.path.splitext(selected_img)[0] + ".txt"
    label_path = os.path.join(images_path.parent.parent, "labels", split, label_file)
    
    st.markdown(f"### Viewing: {selected_img}")
    st.markdown(f"**Dataset Format:** YOLO")
    
    # Visualize image based on selected type
    if vis_type == "Bounding Boxes":
        fig = visualize_yolo_bbox(image_path, label_path, class_names)
        st.pyplot(fig)
    else:  # Instance Segmentation
        fig = visualize_yolo_instance(image_path, label_path, class_names)
        st.pyplot(fig)
    
    # Display label contents
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            label_contents = f.read()
        
        with st.expander("Show Label File Contents"):
            st.code(label_contents)
            
            # Decode label file contents
            lines = label_contents.strip().split('\n')
            decoded = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                    
                cls_id = int(parts[0])
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Unknown class {cls_id}"
                
                if len(parts) == 5:  # bbox
                    decoded.append(f"Class: {cls_name}, Bounding Box: center_x={parts[1]}, center_y={parts[2]}, width={parts[3]}, height={parts[4]}")
                elif len(parts) > 5:  # segmentation
                    num_points = (len(parts) - 1) // 2
                    decoded.append(f"Class: {cls_name}, Segmentation with {num_points} points")
            
            st.markdown("**Decoded Label:**")
            for d in decoded:
                st.markdown(f"- {d}")
    else:
        st.warning(f"No label file found at {label_path}")

def visualize_coco_dataset(dataset_path, split, vis_type):
    """Handle visualization for COCO format datasets"""
    # Load dataset
    dataset_info = load_coco_dataset(dataset_path)
    class_names = dataset_info["class_names"]
    
    # Get data based on split
    if split == "train":
        images_path = dataset_info["train_images_path"]
        annotations_data = dataset_info["train_data"]
    else:
        images_path = dataset_info["val_images_path"]
        annotations_data = dataset_info["val_data"]
    
    if not annotations_data:
        st.error(f"No annotations found for {split} split")
        return
    
    # Get list of images from COCO annotations
    image_list = annotations_data["images"]
    
    if not image_list:
        st.error(f"No images found in {split} split")
        return
    
    # Randomly pick an image or allow user to select
    selection_method = st.sidebar.radio("Image Selection", ["Random", "Choose"])
    
    if selection_method == "Random":
        selected_img_info = random.choice(image_list)
        st.sidebar.button("New Random Image", key="random")
    else:
        image_names = [img["file_name"] for img in image_list]
        selected_img_name = st.sidebar.selectbox("Select Image", image_names)
        selected_img_info = next((img for img in image_list if img["file_name"] == selected_img_name), None)
    
    if not selected_img_info:
        st.error("Failed to get selected image information")
        return
        
    image_id = selected_img_info["id"]
    file_name = selected_img_info["file_name"]
    
    # Validate image path exists
    image_path = os.path.join(images_path, file_name)
    if not os.path.exists(image_path):
        st.error(f"Image file not found: {image_path}")
        return
    
    st.markdown(f"### Viewing: {file_name}")
    st.markdown(f"**Dataset Format:** COCO")
    
    # Visualize image based on selected type
    if vis_type == "Bounding Boxes":
        fig = visualize_coco_bbox(image_path, annotations_data["annotations"], image_id, class_names)
        st.pyplot(fig)
    else:  # Instance Segmentation
        fig = visualize_coco_instance(image_path, annotations_data["annotations"], image_id, class_names)
        st.pyplot(fig)
    
    # Display annotation contents
    image_annotations = [ann for ann in annotations_data["annotations"] if ann["image_id"] == image_id]
    
    with st.expander("Show Annotation Details"):
        if image_annotations:
            for i, ann in enumerate(image_annotations):
                st.markdown(f"**Annotation {i+1}:**")
                
                category_id = ann["category_id"]
                cls_name = class_names[category_id-1] if 0 <= category_id-1 < len(class_names) else f"Class {category_id}"
                
                st.markdown(f"- Class: {cls_name} (ID: {category_id})")
                
                if "bbox" in ann:
                    bbox = ann["bbox"]
                    st.markdown(f"- Bounding Box: x={bbox[0]:.1f}, y={bbox[1]:.1f}, width={bbox[2]:.1f}, height={bbox[3]:.1f}")
                
                if "segmentation" in ann and ann["segmentation"]:
                    seg_points = sum([len(seg)//2 for seg in ann["segmentation"]])
                    st.markdown(f"- Segmentation: {len(ann['segmentation'])} polygon(s) with total {seg_points} points")
                
                st.markdown("---")
        else:
            st.warning(f"No annotations found for this image in the {split} split")

if __name__ == "__main__":
    main() 