import os
import sys
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import random
from loguru import logger

# Fix for Streamlit's file watcher issue with PyTorch
os.environ["STREAMLIT_WATCHDOG_DISABLE"] = "true"

from model import get_instance_segmentation_model

# VOC class names for better visualization
VOC_CLASSES = [
    'background', 'person'
]

# Set up color palette for visualization
def generate_colors(n):
    """Generate n distinct colors for visualization."""
    colors = []
    for i in range(n):
        # Generate vibrant colors with good contrast
        hue = i / n
        saturation = 0.9
        value = 0.9
        
        # Convert HSV to RGB
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        # Return values in 0-1 range instead of 0-255
        colors.append((r+m, g+m, b+m))
    
    return colors

class Inferencer:
    def __init__(self, model_path, num_classes=2, device=None):
        """
        Initialize the inferencer.
        
        Args:
            model_path (str): Path to the trained model checkpoint
            num_classes (int): Number of classes including background
            device (str, optional): Device to run inference on. If None, use CUDA if available.
        """
        if device is None:
            self.device = torch.device('cpu')  # Force CPU for now
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Create model
        self.model = get_instance_segmentation_model(num_classes=num_classes, pretrained=False)
        
        # Load model weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Check if the checkpoint contains a state_dict or is a state_dict itself
            if 'model_state_dict' in checkpoint:
                # If it's a dictionary with a model_state_dict key
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Successfully loaded model from state_dict")
            else:
                # If it's just the state_dict directly
                self.model.load_state_dict(checkpoint)
                logger.info(f"Successfully loaded model directly")
                
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            st.error(f"Failed to load model: {str(e)}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Generate colors for visualization
        self.colors = generate_colors(num_classes)
        
        logger.info(f"Model loaded from {model_path}")
        
        # Print model summary
        logger.info(f"Model structure: {self.model}")
    
    def preprocess_image(self, image):
        """
        Preprocess the input image for inference.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to match training size
        # The model was trained with images resized to have min_size=600, max_size=1000
        w, h = image.size
        min_size = 600
        max_size = 1000
        
        # Calculate new dimensions while preserving aspect ratio
        scale = min_size / min(h, w)
        if h < w:
            new_h, new_w = min_size, int(scale * w)
        else:
            new_h, new_w = int(scale * h), min_size
            
        # Make sure the largest dimension doesn't exceed max_size
        if max(new_h, new_w) > max_size:
            scale = max_size / max(new_h, new_w)
            new_h, new_w = int(scale * new_h), int(scale * new_w)
            
        # Resize the image
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Convert to tensor
        img_tensor = F.to_tensor(image)
        
        # Normalize
        img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        logger.info(f"Preprocessed image from {w}x{h} to {new_w}x{new_h}")
        
        return img_tensor
    
    def predict(self, image, confidence_threshold=0.5):
        """
        Run inference on an image.
        
        Args:
            image (PIL.Image): Input image
            confidence_threshold (float): Confidence threshold for detections
            
        Returns:
            dict: Prediction results
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        img_tensor = img_tensor.to(self.device)
        
        # Get image dimensions
        img_width, img_height = image.size
        
        # Run inference
        try:
            with torch.no_grad():
                prediction = self.model([img_tensor])[0]
                
            logger.info(f"Raw prediction: {prediction}")
            logger.info(f"Number of detections before filtering: {len(prediction['boxes'])}")
            
            # Filter predictions by confidence
            keep = prediction['scores'] > confidence_threshold
            
            filtered_prediction = {
                'boxes': prediction['boxes'][keep].cpu(),
                'labels': prediction['labels'][keep].cpu(),
                'scores': prediction['scores'][keep].cpu(),
                'masks': prediction['masks'][keep].cpu()
            }
            
            # Resize masks if needed
            if 'masks' in filtered_prediction and len(filtered_prediction['masks']) > 0:
                masks = filtered_prediction['masks']
                mask_height, mask_width = masks.shape[2], masks.shape[3]
                
                if mask_height != img_height or mask_width != img_width:
                    logger.info(f"Masks have different dimensions than image. Masks: {masks.shape[2:]} Image: {(img_height, img_width)}")
                    # Note: In a production system, you would resize the masks here
            
            logger.info(f"Number of detections after filtering: {len(filtered_prediction['boxes'])}")
            
            return filtered_prediction
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            st.error(f"Error during prediction: {str(e)}")
            return {'boxes': torch.tensor([]), 'labels': torch.tensor([]), 
                    'scores': torch.tensor([]), 'masks': torch.tensor([])}
    
    def visualize_prediction(self, image, prediction, alpha=0.5):
        """
        Visualize the prediction results on the image.
        
        Args:
            image (PIL.Image): Input image
            prediction (dict): Prediction results
            alpha (float): Transparency of the mask overlay (0-1)
            
        Returns:
            PIL.Image: Visualization image
        """
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Create a copy for drawing
        vis_image = image_np.copy()
        
        # Get prediction components
        boxes = prediction['boxes']
        labels = prediction['labels']
        scores = prediction['scores']
        masks = prediction.get('masks', None)
        
        # Create figure with subplots for different visualizations
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Original image
        axes[0].imshow(image_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # 2. Image with bounding boxes
        axes[1].imshow(image_np)
        for box, label, score in zip(boxes, labels, scores):
            # Get box coordinates
            x1, y1, x2, y2 = box.tolist()
            
            # Create rectangle patch
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, 
                                    linewidth=2, edgecolor='r', facecolor='none')
            axes[1].add_patch(rect)
            
            # Add label and score
            class_name = VOC_CLASSES[label.item()] if label.item() < len(VOC_CLASSES) else f"Class {label.item()}"
            label_text = f"{class_name}: {score.item():.2f}"
            axes[1].text(x1, y1-5, label_text, color='white', fontsize=10,
                        bbox=dict(facecolor='red', alpha=0.7))
        
        axes[1].set_title("Bounding Boxes")
        axes[1].axis('off')
        
        # 3. Image with masks
        axes[2].imshow(image_np)
        if masks is not None and len(masks) > 0:
            # Create a color overlay for all masks
            mask_overlay = np.zeros_like(image_np, dtype=np.uint8)
            
            # Use different colors for different instances
            colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (128, 0, 0),    # Maroon
                (0, 128, 0),    # Green (dark)
                (0, 0, 128),    # Navy
                (128, 128, 0),  # Olive
            ]
            
            # Get image dimensions
            img_height, img_width = image_np.shape[:2]
            
            for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
                # Convert mask to binary numpy array
                mask_np = mask.squeeze().cpu().numpy() > 0.5
                
                # Skip if mask is empty
                if not np.any(mask_np):
                    continue
                
                # Check if mask dimensions match image dimensions
                mask_height, mask_width = mask_np.shape
                if mask_height != img_height or mask_width != img_width:
                    logger.info(f"Resizing mask from {mask_np.shape} to {(img_height, img_width)}")
                    
                    # Resize mask to match image dimensions
                    from PIL import Image as PILImage
                    mask_pil = PILImage.fromarray(mask_np.astype(np.uint8) * 255)
                    mask_pil = mask_pil.resize((img_width, img_height), PILImage.NEAREST)
                    mask_np = np.array(mask_pil) > 0
                
                # Get color for this instance
                color = colors[i % len(colors)]
                
                # Apply mask with color
                for c in range(3):  # RGB channels
                    mask_overlay[:, :, c][mask_np] = color[c]
            
            # Overlay masks on image with alpha blending
            mask_image = image_np.copy()
            mask_image[mask_overlay > 0] = mask_image[mask_overlay > 0] * (1 - alpha) + mask_overlay[mask_overlay > 0] * alpha
            
            axes[2].imshow(mask_image)
        
        axes[2].set_title("Segmentation Masks")
        axes[2].axis('off')
        
        # Add overall title
        plt.suptitle(f"Person Detection Results", fontsize=16)
        plt.tight_layout()
        
        # Convert matplotlib figure to image
        fig.canvas.draw()
        vis_image = np.array(fig.canvas.renderer.buffer_rgba())
        
        # Convert RGBA to RGB
        vis_image = vis_image[:, :, :3]
        
        # Convert numpy array back to PIL Image
        return Image.fromarray(vis_image)

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="Person Detection App",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Person Detection and Segmentation")
    st.write("Upload an image to detect and segment people.")
    
    # Sidebar for model selection and parameters
    st.sidebar.header("Model Settings")
    
    # Model selection
    model_options = {
        "Person Detection Model": "./output_person/best_model.pth",
        "Person Detection Model (Extracted)": "./output_person/model_only.pth"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys())
    )
    
    model_path = model_options[selected_model]
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True)
    
    # Initialize inferencer
    if os.path.exists(model_path):
        inferencer = Inferencer(model_path)
        st.sidebar.success(f"Model loaded: {selected_model}")
    else:
        st.sidebar.error(f"Model file not found: {model_path}")
        st.stop()
    
    # Image upload
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Run inference
        with st.spinner("Running inference..."):
            prediction = inferencer.predict(image, confidence_threshold)
        
        # Display results
        with col2:
            st.subheader("Prediction Results")
            
            # Display number of detections
            num_detections = len(prediction['boxes'])
            st.write(f"Found {num_detections} people with confidence > {confidence_threshold}")
        
        if num_detections > 0:
            # Visualize predictions with enhanced visualization
            vis_image = inferencer.visualize_prediction(image, prediction)
            st.image(vis_image, caption="Prediction Visualization", use_column_width=True)
            
            # Display detailed results in an expandable section
            with st.expander("Detection Details", expanded=False):
                for i in range(num_detections):
                    box = prediction['boxes'][i].tolist()
                    label_idx = prediction['labels'][i].item()
                    label_name = VOC_CLASSES[label_idx] if label_idx < len(VOC_CLASSES) else f"Class {label_idx}"
                    score = prediction['scores'][i].item()
                    
                    st.write(f"Detection {i+1}:")
                    st.write(f"  - Class: {label_name} (ID: {label_idx})")
                    st.write(f"  - Confidence: {score:.4f}")
                    st.write(f"  - Bounding Box: [x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}]")
        else:
            st.write("No people detected. Try lowering the confidence threshold.")
        
        # Debug information
        if debug_mode:
            with st.expander("Debug Information", expanded=False):
                # Show model information
                st.write("### Model Information")
                st.write(f"Model path: {model_path}")
                st.write(f"Device: {inferencer.device}")
                
                # Show raw prediction scores
                if hasattr(inferencer, 'model') and inferencer.model is not None:
                    with torch.no_grad():
                        img_tensor = inferencer.preprocess_image(image).unsqueeze(0).to(inferencer.device)
                        raw_output = inferencer.model(img_tensor)
                        
                        st.write("### Raw Model Output")
                        st.write(f"Output type: {type(raw_output)}")
                        
                        if isinstance(raw_output, list) and len(raw_output) > 0:
                            raw_pred = raw_output[0]
                            st.write("#### Scores Distribution")
                            if 'scores' in raw_pred:
                                scores = raw_pred['scores'].cpu().numpy()
                                st.write(f"Min score: {scores.min() if len(scores) > 0 else 'N/A'}")
                                st.write(f"Max score: {scores.max() if len(scores) > 0 else 'N/A'}")
                                st.write(f"Mean score: {scores.mean() if len(scores) > 0 else 'N/A'}")
                                
                                # Plot score histogram
                                if len(scores) > 0:
                                    fig, ax = plt.subplots()
                                    ax.hist(scores, bins=20)
                                    ax.set_xlabel('Confidence Score')
                                    ax.set_ylabel('Count')
                                    ax.set_title('Distribution of Confidence Scores')
                                    st.pyplot(fig)

if __name__ == "__main__":
    main() 