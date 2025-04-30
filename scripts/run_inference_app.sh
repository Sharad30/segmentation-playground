#!/bin/bash

# Set environment variable to prevent Streamlit from watching torch modules
# This fixes the "RuntimeError: Tried to instantiate class '__path__._path'" error
export STREAMLIT_SERVER_WATCH_MODULES='^(?!torch).*$'

# Run the Streamlit app
echo "Starting YOLOv8 Segmentation Model Inference App..."
streamlit run segmentation/ultralytics/model_inference_app.py 