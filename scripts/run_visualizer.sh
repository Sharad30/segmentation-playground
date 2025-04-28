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

# Check for streamlit installation
if ! command -v streamlit &> /dev/null; then
    print_colored "yellow" "Streamlit is not installed. Installing now..."
    pip install streamlit matplotlib pyyaml opencv-python pillow
    
    if ! command -v streamlit &> /dev/null; then
        print_colored "red" "Failed to install Streamlit. Please install it manually:"
        print_colored "yellow" "pip install streamlit matplotlib pyyaml opencv-python pillow"
        exit 1
    else
        print_colored "green" "Streamlit installed successfully!"
    fi
else
    print_colored "green" "Streamlit is already installed."
fi

# Set path to the visualizer script
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
VISUALIZER_SCRIPT="$SCRIPT_DIR/visualize_dataset.py"

# Make sure the script is executable
chmod +x "$VISUALIZER_SCRIPT"

print_colored "blue" "Starting Dataset Visualizer..."
print_colored "blue" "To view the UI, open your browser to http://localhost:8501"
print_colored "yellow" "Press Ctrl+C to stop the server when done."

# Run the streamlit app
streamlit run "$VISUALIZER_SCRIPT" 