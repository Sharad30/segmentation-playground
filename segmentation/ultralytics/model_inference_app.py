#!/usr/bin/env python3
import streamlit as st
import os
from pathlib import Path
import numpy as np
import yaml
import time
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Wrap potentially problematic imports in try-except blocks
try:
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    st.warning(f"Error importing torch: {e}")
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    st.warning(f"Error importing YOLO: {e}")
    YOLO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except Exception as e:
    st.warning(f"Error importing cv2: {e}")
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception as e:
    st.warning(f"Error importing PIL: {e}")
    PIL_AVAILABLE = False

def find_models():
    """Find all trained models from hyperparameter tuning runs"""
    # Look in runs directory for hyperparameter_tuning folders
    base_runs_dir = Path("runs")
    tuning_dirs = list(base_runs_dir.glob("hyperparameter_tuning*"))
    
    # Also look in the regular train directories
    train_dirs = list(base_runs_dir.glob("train/*"))
    
    all_models = []
    
    # Check hyperparameter tuning runs first
    for tuning_dir in tuning_dirs:
        # Each tuning directory has multiple run directories
        run_dirs = [d for d in tuning_dir.glob("run_*") if d.is_dir()]
        
        for run_dir in run_dirs:
            # Check if this run has a weights directory with models
            weights_dir = run_dir / "weights"
            
            if weights_dir.exists():
                # Look for best.pt and last.pt
                best_model = weights_dir / "best.pt"
                last_model = weights_dir / "last.pt"
                
                # Get metrics if available
                metrics_file = run_dir / "metrics.yaml"
                metrics_info = ""
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = yaml.safe_load(f)
                            box_map = metrics.get('box_map50', 0)
                            mask_map = metrics.get('mask_map50', 0)
                            metrics_info = f" (Box mAP50: {box_map:.3f}, Mask mAP50: {mask_map:.3f})"
                    except:
                        pass
                
                if best_model.exists():
                    model_name = f"{tuning_dir.name}/{run_dir.name}/best{metrics_info}"
                    all_models.append((str(best_model), model_name))
                
                if last_model.exists():
                    model_name = f"{tuning_dir.name}/{run_dir.name}/last"
                    all_models.append((str(last_model), model_name))
    
    # Check regular train directories
    for train_dir in train_dirs:
        weights_dir = train_dir / "weights"
        
        if weights_dir.exists():
            best_model = weights_dir / "best.pt"
            last_model = weights_dir / "last.pt"
            
            if best_model.exists():
                model_name = f"{train_dir.name}/best"
                all_models.append((str(best_model), model_name))
            
            if last_model.exists():
                model_name = f"{train_dir.name}/last"
                all_models.append((str(last_model), model_name))
    
    return all_models

@st.cache_resource
def load_model(model_path):
    """Load the YOLO model with caching"""
    if not TORCH_AVAILABLE or not YOLO_AVAILABLE:
        st.error("Required dependencies (torch/YOLO) not available")
        return None
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_class_names(model_path):
    """Try to load class names from the model or associated dataset.yaml"""
    # First try to load from the model
    if YOLO_AVAILABLE:
        try:
            model = load_model(model_path)
            if model is not None and hasattr(model, 'names') and model.names:
                return model.names
        except Exception as e:
            st.warning(f"Could not load class names from model: {e}")
    
    # If that fails, try to find the dataset.yaml file
    try:
        model_dir = Path(model_path).parent.parent
        yaml_files = list(model_dir.glob("*.yaml"))
        
        for yaml_file in yaml_files:
            if yaml_file.name in ["dataset.yaml", "data.yaml"]:
                with open(yaml_file, 'r') as f:
                    data_config = yaml.safe_load(f)
                    if 'names' in data_config:
                        return data_config['names']
    except Exception as e:
        st.warning(f"Error loading class names from YAML: {e}")
    
    # Default to generic class names if nothing found
    return {0: "Object"}

def run_inference(model_path, image, conf_threshold=0.25, device='0'):
    """Run inference using the selected model"""
    if not TORCH_AVAILABLE or not YOLO_AVAILABLE:
        st.error("Required dependencies (torch/YOLO) not available")
        return None, 0
    
    try:
        # Load the model
        model = load_model(model_path)
        if model is None:
            return None, 0
        
        # Ensure device is set correctly for GPU inference
        if torch.cuda.is_available() and device != 'cpu':
            device = '0' if device == '' else device
            st.info(f"Using GPU device: {device}")
        else:
            device = 'cpu'
            st.info("Using CPU for inference")
            
        # Run inference
        start_time = time.time()
        results = model.predict(
            image, 
            conf=conf_threshold,
            device=device,
            verbose=False
        )
        inference_time = time.time() - start_time
        
        return results, inference_time
    except Exception as e:
        st.error(f"Error running inference: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, 0

def display_results(results, class_names, image):
    """Display segmentation results with color-coded masks"""
    if results is None or len(results) == 0:
        st.warning("No objects detected in the image.")
        return
    
    # Get the first result (only one image was processed)
    result = results[0]
    
    # Convert PIL Image to numpy array if needed
    if PIL_AVAILABLE and isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Display the original image first
    ax.imshow(image_np)
    
    # Generate colors for each class
    num_classes = len(class_names)
    colors = plt.cm.get_cmap('tab10', num_classes)
    
    # For legend
    legend_patches = []
    used_classes = set()
    
    # Process masks if available
    if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
        try:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # Get class ID and confidence
                class_id = int(box[5])
                confidence = box[4]
                
                # Get color for this class
                color_tuple = colors(class_id % num_classes)
                
                # Create a boolean mask
                mask_bool = mask > 0
                
                # Apply colored mask with transparency
                # Create a transparent overlay
                overlay = ax.imshow(
                    np.ones_like(image_np[:,:,0]), 
                    alpha=np.where(mask_bool, 0.5, 0),
                    cmap=plt.cm.colors.ListedColormap([color_tuple[:3]]),
                    extent=(0, image_np.shape[1], image_np.shape[0], 0)
                )
                
                # Get class name
                class_name = class_names.get(class_id, f"Class {class_id}")
                
                # Add to legend if not already added
                if class_id not in used_classes:
                    patch = mpatches.Patch(color=color_tuple, label=f"{class_name}")
                    legend_patches.append(patch)
                    used_classes.add(class_id)
                    
                # Draw bounding box and label
                x1, y1, x2, y2 = box[:4]
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color_tuple, linewidth=2)
                ax.add_patch(rect)
                
                # Add label with confidence
                ax.text(
                    x1, y1-5, 
                    f"{class_name} {confidence:.2f}", 
                    color='white', 
                    fontsize=10, 
                    backgroundcolor=color_tuple
                )
        except Exception as e:
            st.error(f"Error processing masks: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Or just boxes if masks aren't available
    elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        try:
            boxes = result.boxes.data.cpu().numpy()
            
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box[5])
                confidence = box[4]
                
                # Get color for this class
                color = colors(class_id % num_classes)
                
                # Get class name
                class_name = class_names.get(class_id, f"Class {class_id}")
                
                # Add to legend if not already added
                if class_id not in used_classes:
                    patch = mpatches.Patch(color=color, label=f"{class_name}")
                    legend_patches.append(patch)
                    used_classes.add(class_id)
                    
                # Draw bounding box and label
                x1, y1, x2, y2 = box[:4]
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2)
                ax.add_patch(rect)
                
                # Add label with confidence
                ax.text(
                    x1, y1-5, 
                    f"{class_name} {confidence:.2f}", 
                    color='white', 
                    fontsize=10, 
                    backgroundcolor=color
                )
        except Exception as e:
            st.error(f"Error processing boxes: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Add legend if we have patches
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right')
    
    # Remove axis
    ax.axis('off')
    
    # Display the figure
    st.pyplot(fig)
    
    # Display detection info
    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        try:
            boxes = result.boxes.data.cpu().numpy()
            st.write(f"Number of objects detected: {len(boxes)}")
            
            # Create a dataframe of detections
            if len(boxes) > 0:
                detection_data = []
                for box in boxes:
                    class_id = int(box[5])
                    class_name = class_names.get(class_id, f"Class {class_id}")
                    confidence = box[4]
                    x1, y1, x2, y2 = box[:4]
                    
                    detection_data.append({
                        "Class": class_name,
                        "Confidence": f"{confidence:.3f}",
                        "Box": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                    })
                
                if detection_data:
                    st.table(detection_data)
        except Exception as e:
            st.error(f"Error displaying detection data: {e}")

def main():
    try:
        st.set_page_config(page_title="YOLOv8 Segmentation Model Inference", layout="wide")
        
        st.title("YOLOv8 Segmentation Model Inference")
        st.write("Select a trained model and upload an image to run segmentation inference")
        
        # Check dependencies
        if not TORCH_AVAILABLE:
            st.error("PyTorch is not available. Please install it to use this app.")
            return
            
        if not YOLO_AVAILABLE:
            st.error("Ultralytics YOLO is not available. Please install it to use this app.")
            return
            
        if not PIL_AVAILABLE:
            st.error("PIL/Pillow is not available. Please install it to use this app.")
            return
        
        # Check CUDA
        cuda_info = "CUDA is not available"
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                cuda_info = f"CUDA available: {torch.cuda.get_device_name(0)}"
                st.sidebar.success(cuda_info)
            else:
                st.sidebar.warning("CUDA not available. Using CPU for inference.")
        
        # List available models
        all_models = find_models()
        
        if not all_models:
            st.error("No trained models found in the runs directory. Please run training or hyperparameter tuning first.")
            return
        
        # Model selection
        model_options = [f"{name}" for _, name in all_models]
        selected_model_name = st.selectbox("Select a model", model_options)
        
        # Get the model path for the selected name
        selected_model_path = next((path for path, name in all_models if name == selected_model_name), None)
        
        if selected_model_path:
            st.write(f"Model path: `{selected_model_path}`")
            
            # Load class names for the selected model
            class_names = load_class_names(selected_model_path)
            
            # Settings
            col1, col2 = st.columns(2)
            with col1:
                conf_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
            with col2:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    device_options = ['0'] + [str(i) for i in range(1, torch.cuda.device_count())] + ['cpu']
                    device = st.selectbox("Device", device_options, index=0)
                    st.write(cuda_info)
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
                            results, inference_time = run_inference(
                                selected_model_path, 
                                image, 
                                conf_threshold,
                                device
                            )
                        
                        if results is not None:
                            # Display inference time
                            st.success(f"Inference completed in {inference_time:.3f} seconds")
                            
                            # Display results
                            st.subheader("Results")
                            display_results(results, class_names, image)
                        else:
                            st.error("Inference failed. Check the error messages above.")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 