#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
from loguru import logger

# Add the parent directory to the path so we can use absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from segmentation.pytorch.train import main as train_main

def parse_args():
    parser = argparse.ArgumentParser(description='Run UNet training for semantic segmentation')
    parser.add_argument('--config', type=str, default=None, help='Path to config yaml file')
    return parser.parse_args()

def main():
    # Get arguments
    args = parse_args()
    
    # Set default config file path if not provided
    if args.config is None:
        # Use the config file in the same directory
        args.config = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    # Initialize logger
    logger.info(f"Starting UNet training using config: {args.config}")
    
    # Setup arguments for train_main
    sys.argv = [sys.argv[0], '--config', args.config]
    
    # Run training
    train_main()

if __name__ == '__main__':
    main() 