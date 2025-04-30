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

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from segmentation.pytorch.model import create_model

def find_models():
    """Find all trained models from hyperparameter tuning runs"""
    # Look in artifacts directory for hyperparameter tuning folders
    base_dir = Path("artifacts/pytorch")
    tuning_dirs = list(base_dir.glob("hyperparameter_tuning*"))
    
    all_models = []
    
    # Check hyperparameter tuning runs
    for tuning_dir in tuning_dirs:
        # Each tuning directory has multiple run directories
        run_dirs = [d for d in tuning_dir.glob("run_*") if d.is_dir()]
        
        for run_dir in run_dirs:
            # Check if this run has a best checkpoint
            checkpoint_path = run_dir / "output" / "checkpoint_best.pth"
            
            if checkpoint_path.exists():
                # Get metrics if available
                metrics_file = run_dir / "metrics.yaml"
                metrics_info = ""
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = yaml.safe_load(f)
                            val_dice = metrics.get('val_dice', 0)
                            val_pixel_acc = metrics.get('val_pixel_acc', 0)
                            metrics_info = f" (Dice: {val_dice:.3f}, Pixel Acc: {val_pixel_acc:.3f})"
                    except:
                        pass
                
                model_name = f"{tuning_dir.name}/{run_dir.name}/best{metrics_info}"
                all_models.append((str(checkpoint_path), model_name))
    
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
        model = create_model(config)
        
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

def main():
    try:
        st.set_page_config(page_title="UNet Semantic Segmentation Inference", layout="wide")
        
        st.title("UNet Semantic Segmentation Inference")
        st.write("Select a trained model and upload an image to run semantic segmentation")
        
        # Check CUDA
        if torch.cuda.is_available():
            device = "cuda"
            cuda_info = f"CUDA available: {torch.cuda.get_device_name(0)}"
            st.sidebar.success(cuda_info)
        else:
            device = "cpu"
            st.sidebar.warning("CUDA not available. Using CPU for inference.")
        
        # List available models
        all_models = find_models()
        
        if not all_models:
            st.error("No trained models found in the artifacts directory. Please run training or hyperparameter tuning first.")
            return
        
        # Model selection
        model_options = [name for _, name in all_models]
        selected_model_name = st.selectbox("Select a model", model_options)
        
        # Get the model path for the selected name
        selected_model_path = next((path for path, name in all_models if name == selected_model_name), None)
        
        if selected_model_path:
            st.write(f"Model path: `{selected_model_path}`")
            
            # Load model
            with st.spinner("Loading model..."):
                model, class_names, config = load_model(selected_model_path, device)
            
            if model is None:
                st.error("Failed to load model.")
                return
                
            # Display model info
            st.write(f"Model classes: {len(class_names)}")
            
            # Settings
            col1, col2 = st.columns(2)
            with col1:
                input_size = st.slider("Input Size", min_value=256, max_value=1024, value=config.get('input_size', 512), step=64)
            with col2:
                if torch.cuda.is_available():
                    device_options = ['cuda', 'cpu']
                    device = st.selectbox("Device", device_options, index=0)
                else:
                    device = 'cpu'
                    st.warning("CUDA not available. Using CPU for inference.")
            
            # Image upload
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            
            use_camera = st.checkbox("Or use camera input")
            camera_image = None
            
            if use_camera:
                camera_image = st.camera_input("Take a photo")
            
            # When image is uploaded or camera used
            if uploaded_file is not None or camera_image is not None:
                try:
                    image_source = camera_image if uploaded_file is None else uploaded_file
                    
                    # Load image
                    image = Image.open(image_source)
                    
                    # Display original image
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                    
                    # Run inference
                    if st.button("Run Inference"):
                        with st.spinner("Running inference..."):
                            mask, inference_time = run_inference(
                                model, 
                                image, 
                                device,
                                input_size
                            )
                        
                        if mask is not None:
                            # Display inference time
                            st.success(f"Inference completed in {inference_time:.3f} seconds")
                            
                            # Display results
                            st.subheader("Results")
                            display_results(image, mask, class_names)
                        else:
                            st.error("Inference failed. Check the error messages above.")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    logger.error(f"Error processing image: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 