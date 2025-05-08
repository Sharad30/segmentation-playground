#!/usr/bin/env python3
import streamlit as st
import os
import sys
from pathlib import Path
import numpy as np
import yaml
import time
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from loguru import logger
import json

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from segmentation.pytorch.model import ResNetUNet, create_model

# Set up logger
logger.add("unet_inference.log", rotation="100 MB")

def find_models():
    """Find all trained UNet models in the artifacts directory"""
    # Look in artifacts directory for output folders
    base_dir = Path("artifacts/pytorch")
    
    # Find checkpoint files recursively in all subdirectories
    checkpoint_paths = list(base_dir.rglob("checkpoint_*.pth"))
    
    all_models = []
    
    for path in checkpoint_paths:
        # Get the run directory name (parent of output directory)
        run_dir = path.parent.parent if path.parent.name == "output" else path.parent
        
        # Determine model type (best or latest)
        model_type = "best" if "best" in path.name else "latest"
        
        # Try to extract metrics if available
        metrics_info = ""
        try:
            # First try to load metrics from metrics.yaml
            metrics_path = path.parent / "metrics.yaml"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = yaml.safe_load(f)
                    if 'val_dice' in metrics:
                        metrics_info = f" (Dice: {metrics['val_dice']:.3f})"
            else:
                # Fallback to checkpoint file
                checkpoint = torch.load(path, map_location='cpu')
                if 'val_dice' in checkpoint:
                    metrics_info = f" (Dice: {checkpoint['val_dice']:.3f})"
                elif 'config' in checkpoint and 'val_dice' in checkpoint.get('metrics', {}):
                    metrics_info = f" (Dice: {checkpoint['metrics']['val_dice']:.3f})"
        except Exception as e:
            logger.error(f"Error loading metrics for {path}: {e}")
        
        # Create model name using run directory name
        model_name = f"{run_dir.name}/{model_type}{metrics_info}"
        all_models.append((str(path), model_name))
    
    # Sort models by name
    all_models.sort(key=lambda x: x[1])
    
    return all_models

@st.cache_resource
def load_model(model_path, device='cuda'):
    """Load the UNet model with caching"""
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get config from checkpoint
        config = checkpoint.get('config', {})
        
        # Create model
        model = ResNetUNet(
            n_channels=config.get('n_channels', 3),
            n_classes=config.get('num_classes', 21),
            bilinear=config.get('bilinear', True)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Get class names if available
        class_names = {}
        try:
            data_path = config.get('data_dir')
            if data_path:
                # Try to load class names from data.yaml in the dataset directory
                yaml_path = os.path.join(data_path, 'dataset.yaml')
                if os.path.exists(yaml_path):
                    with open(yaml_path, 'r') as f:
                        data_yaml = yaml.safe_load(f)
                        names = data_yaml.get('names', [])
                        for i, name in enumerate(names):
                            class_names[i + 1] = name  # +1 because 0 is background
                    class_names[0] = 'background'
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            
        # If class names couldn't be loaded, use default VOC class names
        if not class_names:
            class_names = {
                0: 'background',
                1: 'aeroplane',
                2: 'bicycle',
                3: 'bird',
                4: 'boat',
                5: 'bottle',
                6: 'bus',
                7: 'car',
                8: 'cat',
                9: 'chair',
                10: 'cow',
                11: 'diningtable',
                12: 'dog',
                13: 'horse',
                14: 'motorbike',
                15: 'person',
                16: 'pottedplant',
                17: 'sheep',
                18: 'sofa',
                19: 'train',
                20: 'tvmonitor'
            }
            
        return model, class_names, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        logger.error(f"Error loading model: {e}")
        return None, {}, {}

def preprocess_image(image, input_size=512):
    """Preprocess image for inference"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    original_size = image.size
    image = image.resize((input_size, input_size), Image.BILINEAR)
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_size

def run_inference(model, image, device='cuda', input_size=512):
    """Run inference using the UNet model"""
    try:
        # Preprocess image
        image_tensor, original_size = preprocess_image(image, input_size)
        image_tensor = image_tensor.to(device)
        
        # Run inference with timer
        start_time = time.time()
        with torch.no_grad():
            output = model(image_tensor)
        inference_time = time.time() - start_time
        
        # Get class predictions
        output = F.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]  # [H, W]
        
        # Resize prediction back to original size
        pred_resized = cv2.resize(
            pred.astype(np.uint8), 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        return pred_resized, inference_time
    except Exception as e:
        st.error(f"Error running inference: {e}")
        logger.error(f"Error running inference: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, 0

def create_colored_mask(mask, class_names):
    """Create a colored mask for visualization"""
    num_classes = len(class_names)
    
    # Create a colormap
    colormap = plt.cm.get_cmap('tab20', num_classes)
    
    # Initialize colored mask
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Fill each class with its color
    for class_id in range(num_classes):
        if class_id == 0:  # Skip background
            continue
            
        # Find pixels of this class
        indices = mask == class_id
        if not np.any(indices):
            continue
            
        # Get color for this class
        color = np.array(colormap(class_id)[:3]) * 255
        
        # Set color for these pixels
        colored_mask[indices] = color
    
    return colored_mask

def display_results(image, mask, class_names):
    """Display segmentation results with color-coded masks"""
    if mask is None:
        st.warning("No segmentation mask produced.")
        return
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Create a new figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display original image
    ax1.imshow(image_np)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Display colored mask
    colored_mask = create_colored_mask(mask, class_names)
    ax2.imshow(colored_mask)
    ax2.set_title("Segmentation Mask")
    ax2.axis('off')
    
    # Display overlay
    overlay = image_np.copy()
    alpha = 0.5
    mask_indices = mask > 0  # non-background pixels
    if np.any(mask_indices):
        overlay[mask_indices] = (
            alpha * overlay[mask_indices] + (1 - alpha) * colored_mask[mask_indices]
        ).astype(np.uint8)
    
    ax3.imshow(overlay)
    ax3.set_title("Overlay")
    ax3.axis('off')
    
    # Create legend for classes
    legend_patches = []
    used_classes = set()
    
    # Get unique classes in the mask
    unique_classes = np.unique(mask)
    
    # Create a colormap
    colormap = plt.cm.get_cmap('tab20', len(class_names))
    
    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue
            
        class_name = class_names.get(class_id, f"Class {class_id}")
        color = colormap(class_id)[:3]
        
        # Add to legend
        patch = mpatches.Patch(color=color, label=class_name)
        legend_patches.append(patch)
        used_classes.add(class_id)
    
    # Add legend if we have patches
    if legend_patches:
        fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=min(5, len(legend_patches)))
    
    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display class statistics
    st.subheader("Class Statistics")
    
    if len(used_classes) == 0:
        st.warning("No classes detected in the image.")
    else:
        # Count pixels per class
        class_stats = []
        total_pixels = mask.size
        
        for class_id in sorted(used_classes):
            class_name = class_names.get(class_id, f"Class {class_id}")
            pixel_count = np.sum(mask == class_id)
            percentage = (pixel_count / total_pixels) * 100
            
            class_stats.append({
                "Class": class_name,
                "Pixels": pixel_count,
                "Percentage": f"{percentage:.2f}%"
            })
        
        # Display as table
        st.table(class_stats)

def load_validation_images_and_masks():
    """Load validation images and their corresponding masks from VOC-COCO semantic dataset"""
    val_images_dir = "/home/ubuntu/sharad/segmentation-playground/datasets/processed/voc_coco_semantic/images/val"
    val_annotations_file = "/home/ubuntu/sharad/segmentation-playground/datasets/processed/voc_coco_semantic/annotations/instances_val.json"
    
    # Get list of validation images
    image_files = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    
    # Create a list of tuples (image_path, image_id)
    image_paths = []
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        image_id = os.path.splitext(image_name)[0]  # Remove .jpg extension
        image_paths.append((image_path, image_id))
    
    return image_paths

def load_coco_annotations(image_id):
    """Load COCO format annotations for a specific image"""
    annotations_file = "/home/ubuntu/sharad/segmentation-playground/datasets/processed/voc_coco_semantic/annotations/instances_val.json"
    
    try:
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Find image info
        image_info = None
        for img in coco_data['images']:
            if img['file_name'].startswith(image_id):
                image_info = img
                break
        
        if image_info is None:
            return None
            
        # Get all annotations for this image
        image_annotations = []
        for ann in coco_data['annotations']:
            if ann['image_id'] == image_info['id']:
                image_annotations.append(ann)
        
        return image_info, image_annotations
        
    except Exception as e:
        st.error(f"Error loading annotations: {e}")
        return None

def create_mask_from_coco(image_shape, annotations):
    """Create a segmentation mask from COCO format annotations"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    try:
        for ann in annotations:
            category_id = ann['category_id']
            segmentation = ann['segmentation']
            
            # Convert segmentation to binary mask
            for seg in segmentation:
                # Convert polygon to numpy array
                poly = np.array(seg).reshape(-1, 2)
                
                # Convert to integer coordinates
                poly = poly.astype(np.int32)
                
                # Fill polygon with category ID
                cv2.fillPoly(mask, [poly], category_id)
    
    except Exception as e:
        st.error(f"Error creating mask: {e}")
    
    return mask

def display_validation_results(image, actual_mask, pred_mask, class_names):
    """Display original image, actual mask, and predicted mask side by side"""
    # Create three columns for display with equal width
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Resize images for display
    display_height = 250  # Reduced from 300 to 250
    aspect_ratio = image.shape[1] / image.shape[0]
    display_width = int(display_height * aspect_ratio)
    display_image = cv2.resize(image, (display_width, display_height))
    
    # Resize masks to match display size
    display_actual_mask = cv2.resize(actual_mask, (display_width, display_height), interpolation=cv2.INTER_NEAREST)
    display_pred_mask = cv2.resize(pred_mask, (display_width, display_height), interpolation=cv2.INTER_NEAREST)
    
    # Display original image
    with col1:
        st.subheader("Original Image")
        st.image(display_image, width=display_width)
    
    # Calculate figure size
    fig_size = 2.5  # Fixed small figure size
    
    # Display actual mask
    with col2:
        st.subheader("Ground Truth")
        fig, ax = plt.subplots(figsize=(fig_size * aspect_ratio, fig_size))
        
        # Remove padding and margins
        plt.subplots_adjust(left=0, right=0.85, bottom=0, top=1, wspace=0, hspace=0)
        
        ax.imshow(display_image)
        
        # Create colored mask for visualization
        colored_mask = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        colormap = plt.cm.get_cmap('tab20', len(class_names))
        
        for class_id in range(len(class_names)):
            if class_id == 0:  # Skip background
                continue
            indices = display_actual_mask == class_id
            if np.any(indices):
                color = np.array(colormap(class_id)[:3]) * 255
                colored_mask[indices] = color
        
        ax.imshow(colored_mask, alpha=0.5)
        ax.axis('off')
        st.pyplot(fig)
    
    # Display predicted mask
    with col3:
        st.subheader("Prediction")
        fig, ax = plt.subplots(figsize=(fig_size * aspect_ratio, fig_size))
        
        # Remove padding and margins
        plt.subplots_adjust(left=0, right=0.85, bottom=0, top=1, wspace=0, hspace=0)
        
        ax.imshow(display_image)
        
        # Create colored mask for predictions
        colored_pred_mask = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        for class_id in range(len(class_names)):
            if class_id == 0:  # Skip background
                continue
            indices = display_pred_mask == class_id
            if np.any(indices):
                color = np.array(colormap(class_id)[:3]) * 255
                colored_pred_mask[indices] = color
        
        ax.imshow(colored_pred_mask, alpha=0.5)
        ax.axis('off')
        st.pyplot(fig)
    
    # Display class statistics
    st.subheader("Class Statistics")
    
    # Count pixels per class in both actual and predicted masks
    stats = []
    total_pixels = actual_mask.size
    
    for class_id in range(len(class_names)):
        if class_id == 0:  # Skip background
            continue
            
        class_name = class_names.get(class_id, f"Class {class_id}")
        actual_count = np.sum(actual_mask == class_id)
        pred_count = np.sum(pred_mask == class_id)
        actual_percent = (actual_count / total_pixels) * 100
        pred_percent = (pred_count / total_pixels) * 100
        
        if actual_count > 0 or pred_count > 0:
            stats.append({
                "Class": class_name,
                "Ground Truth Pixels": actual_count,
                "Ground Truth %": f"{actual_percent:.2f}%",
                "Predicted Pixels": pred_count,
                "Predicted %": f"{pred_percent:.2f}%"
            })
    
    if stats:
        st.table(stats)
    else:
        st.warning("No classes detected in either ground truth or predicted masks.")

def main():
    try:
        st.set_page_config(page_title="UNet Semantic Segmentation Inference", layout="wide")
        
        st.title("UNet Semantic Segmentation Inference")
        st.write("Select a trained model and upload an image to run semantic segmentation")
        
        # Find available models
        available_models = find_models()
        if not available_models:
            st.error("No trained models found in artifacts directory")
            return
            
        # Model selection
        selected_model_path, selected_model_name = st.selectbox(
            "Select a model",
            available_models,
            format_func=lambda x: x[1]
        )
        
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, class_names, config = load_model(selected_model_path, device)
        if model is None:
            return
            
        # Add tabs for different input methods
        tab1, tab2 = st.tabs(["Upload Image", "Random Validation Image"])
        
        with tab1:
            # File uploader
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                try:
                    # Load and preprocess image
                    image = Image.open(uploaded_file)
                    image_array = np.array(image)
                    
                    # Ensure image is RGB
                    if len(image_array.shape) == 2:  # Grayscale
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                    elif image_array.shape[2] == 4:  # RGBA
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                    
                    # Resize image if needed
                    if config.get('input_size'):
                        input_size = config['input_size']
                        image_array = cv2.resize(image_array, (input_size, input_size))
                    
                    # Normalize and convert to tensor
                    input_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float() / 255.0
                    input_tensor = input_tensor.unsqueeze(0)
                    input_tensor = input_tensor.to(device)
                    
                    # Make prediction
                    with torch.no_grad():
                        output = model(input_tensor)
                        output = F.softmax(output, dim=1)
                        pred_mask = output.argmax(1).squeeze().cpu().numpy()
                    
                    # Resize images for display
                    display_height = 250  # Reduced from 300 to 250
                    aspect_ratio = image_array.shape[1] / image_array.shape[0]
                    display_width = int(display_height * aspect_ratio)
                    display_image = cv2.resize(image_array, (display_width, display_height))
                    
                    # Resize prediction mask to match display size
                    display_mask = cv2.resize(pred_mask, (display_width, display_height), interpolation=cv2.INTER_NEAREST)
                    
                    # Display images side by side using two columns with equal width
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Original Image")
                        st.image(display_image, width=display_width)
                    
                    with col2:
                        st.subheader("Prediction")
                        # Create figure with matching size
                        fig_size = 2.5  # Fixed small figure size
                        fig, ax = plt.subplots(figsize=(fig_size * aspect_ratio, fig_size))
                        
                        # Remove padding and margins
                        plt.subplots_adjust(left=0, right=0.85, bottom=0, top=1, wspace=0, hspace=0)
                        
                        # Display the resized image
                        ax.imshow(display_image)
                        
                        # Create colormap and colored mask
                        colormap = plt.cm.get_cmap('tab20', len(class_names))
                        colored_mask = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                        
                        # Create legend patches
                        legend_patches = []
                        
                        for class_id in range(len(class_names)):
                            if class_id == 0:  # Skip background
                                continue
                            indices = display_mask == class_id
                            if np.any(indices):
                                color = colormap(class_id)[:3]
                                colored_mask[indices] = np.array(color) * 255
                                
                                # Add to legend
                                patch = mpatches.Patch(
                                    color=color, 
                                    label=f"{class_names[class_id]}"
                                )
                                legend_patches.append(patch)
                        
                        # Show mask overlay
                        ax.imshow(colored_mask, alpha=0.5)
                        
                        # Add legend if we have any classes
                        if legend_patches:
                            # Move legend outside the plot to prevent overlap
                            ax.legend(
                                handles=legend_patches,
                                bbox_to_anchor=(1.15, 1),
                                loc='upper left',
                                fontsize=8
                            )
                        
                        ax.axis('off')
                        st.pyplot(fig)
                    
                    # Display class statistics below the images
                    st.subheader("Class Statistics")
                    class_stats = []
                    total_pixels = pred_mask.size
                    
                    for class_id in range(len(class_names)):
                        if class_id == 0:  # Skip background
                            continue
                        indices = pred_mask == class_id
                        if np.any(indices):
                            pixel_count = np.sum(indices)
                            percentage = (pixel_count / total_pixels) * 100
                            class_stats.append({
                                "Class": class_names[class_id],
                                "Pixels": pixel_count,
                                "Percentage": f"{percentage:.2f}%"
                            })
                    
                    if class_stats:
                        st.table(class_stats)
                    else:
                        st.warning("No classes detected in the image")
                
                except Exception as e:
                    st.error(f"Error processing uploaded image: {e}")
                    logger.error(f"Error processing uploaded image: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        with tab2:
            # Load validation images
            image_paths = load_validation_images_and_masks()
            
            if not image_paths:
                st.error("No validation images found")
                return
            
            # Add checkbox for random image selection
            if st.checkbox("Load Random Validation Image"):
                try:
                    # Randomly select an image
                    image_path, image_id = image_paths[np.random.randint(len(image_paths))]
                    
                    # Load image and annotations
                    image = cv2.imread(image_path)
                    if image is None:
                        st.error(f"Failed to load image: {image_path}")
                        return
                        
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Load COCO annotations
                    result = load_coco_annotations(image_id)
                    if result is None:
                        st.error(f"Failed to load annotations for image: {image_id}")
                        return
                        
                    image_info, annotations = result
                    
                    # Create ground truth mask
                    actual_mask = create_mask_from_coco(image.shape[:2], annotations)
                    
                    # Resize if needed
                    if config.get('input_size'):
                        input_size = config['input_size']
                        image = cv2.resize(image, (input_size, input_size))
                        actual_mask = cv2.resize(actual_mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
                    
                    # Prepare input tensor
                    input_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                    input_tensor = input_tensor.unsqueeze(0)
                    input_tensor = input_tensor.to(device)
                    
                    # Make prediction
                    with torch.no_grad():
                        output = model(input_tensor)
                        output = F.softmax(output, dim=1)
                        pred_mask = output.argmax(1).squeeze().cpu().numpy()
                    
                    # Display results
                    display_validation_results(image, actual_mask, pred_mask, class_names)
                    
                except Exception as e:
                    st.error(f"Error processing validation image: {e}")
                    logger.error(f"Error processing validation image: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in main: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 