#!/usr/bin/env python3
import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
import shutil
import datetime
import time
import cv2

# Import skimage's measure module with error handling
try:
    from skimage import measure
except ImportError:
    print("Warning: Could not import skimage.measure. Attempting alternative import paths...")
    try:
        # Try scikit-image instead of skimage
        from scikit_image import measure
    except ImportError:
        print("Error: Could not import skimage.measure or scikit_image.measure")
        print("Please install scikit-image: pip install scikit-image")
        print("If that fails, try: pip install scikit-image==0.19.3")
        measure = None

from shapely.geometry import Polygon, MultiPolygon

# Check if required modules are available
def check_requirements():
    missing = []
    if measure is None:
        missing.append("scikit-image (module 'measure')")
    
    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}")
        print("Please install the missing dependencies and try again.")
        return False
    return True

def get_categories(annotations_dir):
    """Extract all unique categories from VOC XML annotations"""
    categories = set()
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith('.xml'):
            continue
        
        xml_path = os.path.join(annotations_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            categories.add(name)
    
    # Sort categories to ensure consistent category IDs
    categories = sorted(list(categories))
    return {cat: i+1 for i, cat in enumerate(categories)}

def create_sub_mask(mask_image, obj_id):
    """Create a binary mask for a specific object ID"""
    mask = np.array(mask_image)
    return mask == obj_id

def create_sub_masks(mask_path, verbose=False, seg_type='instance'):
    """Create sub-masks for each object in the segmentation mask"""
    if verbose:
        print(f"Processing mask: {mask_path}")
    
    mask_img = Image.open(mask_path)
    mask_img = mask_img.convert("P")
    mask = np.array(mask_img)
    
    # Get unique color values in mask (each color is an object)
    obj_ids = np.unique(mask)
    
    # Remove background (0)
    obj_ids = obj_ids[obj_ids > 0]
    
    if verbose:
        print(f"Found {len(obj_ids)} unique object IDs in mask: {obj_ids}")
    
    sub_masks = {}
    for obj_id in obj_ids:
        if verbose:
            print(f"Creating sub-mask for object ID {obj_id}")
        sub_masks[obj_id] = create_sub_mask(mask, obj_id)
    
    return sub_masks

def create_polygon_from_mask(mask, tolerance=0.5, verbose=False):
    """Convert binary mask to polygon"""
    if verbose:
        print(f"Creating polygon from mask with shape {mask.shape}")
    
    # Ensure mask is binary
    binary_mask = mask.astype(np.uint8)
    
    # Find contours using OpenCV instead of skimage for more reliable results
    try:
        # Use CHAIN_APPROX_NONE to get all contour points instead of simplified contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if verbose:
            print(f"Found {len(contours)} contours using OpenCV")
    except:
        # Fallback to skimage if OpenCV fails
        contours = measure.find_contours(binary_mask, 0.5)
        if verbose:
            print(f"Found {len(contours)} contours using skimage")
    
    # Convert contours to polygons
    polygons = []
    
    if len(contours) == 0 and verbose:
        print("Warning: No contours found in mask")
    
    for contour in contours:
        # Skip tiny contours - these often cause problems
        if len(contour) < 4:
            if verbose:
                print(f"Skipping contour with only {len(contour)} points")
            continue
        
        # OpenCV contours are already in the format we need
        if isinstance(contour, np.ndarray) and len(contour.shape) == 3:
            # OpenCV contour
            contour = contour.reshape(-1, 2)
        else:
            # skimage contour needs to be reshaped
            contour = np.array(contour).reshape(-1, 2)
        
        # Calculate object size for adaptive tolerance
        try:
            contour_area = cv2.contourArea(contour.reshape(-1, 1, 2))
        except:
            # Fallback if reshaping fails
            if verbose:
                print(f"Error calculating contour area, using approximation")
            contour_area = len(contour) * 2  # Simple approximation
        
        # Skip very small contours
        if contour_area < 50:
            if verbose:
                print(f"Skipping tiny contour with area {contour_area}")
            continue
        
        # Use a lower tolerance for smaller objects and scale with area
        adaptive_tolerance = min(tolerance, max(0.1, contour_area / 10000))
        
        if verbose:
            print(f"Contour has {len(contour)} points with area {contour_area}, using tolerance {adaptive_tolerance}")
        
        # Convert to Shapely polygon and simplify
        try:
            # For OpenCV contours, we need to ensure x is first, y is second for Shapely
            if len(contour) >= 3:  # Need at least 3 points for a polygon
                # Create a valid polygon
                poly = Polygon(contour).buffer(0)  # buffer(0) fixes self-intersections
                if poly.is_valid:
                    poly = poly.simplify(tolerance=adaptive_tolerance, preserve_topology=True)
                else:
                    if verbose:
                        print(f"Invalid polygon, trying to fix with buffer(0)")
                    continue
            else:
                if verbose:
                    print(f"Not enough points for polygon: {len(contour)}")
                continue
                
            # Skip invalid or empty polygons
            if poly.is_empty or not poly.is_valid:
                if verbose:
                    print("Skipping invalid or empty polygon")
                continue
            
            # Handle multi-part polygons
            if isinstance(poly, MultiPolygon):
                for p in poly.geoms:
                    if p.exterior and len(p.exterior.coords) >= 3:
                        polygons.append(p.exterior.coords)
                        if verbose:
                            print(f"Added multipolygon part with {len(p.exterior.coords)} points")
            else:
                if poly.exterior and len(poly.exterior.coords) >= 3:
                    polygons.append(poly.exterior.coords)
                    if verbose:
                        print(f"Added polygon with {len(poly.exterior.coords)} points")
        except Exception as e:
            if verbose:
                print(f"Error processing contour: {str(e)}")
            continue
    
    # If no valid polygons were created but we had contours, try a simpler approach
    if len(polygons) == 0 and len(contours) > 0:
        if verbose:
            print("No valid polygons created, trying simpler approach")
        
        # Try a much simpler approach - just use the largest contour directly
        largest_contour_idx = 0
        largest_area = 0
        
        for i, contour in enumerate(contours):
            if isinstance(contour, np.ndarray) and len(contour.shape) == 3:
                # OpenCV contour
                contour_flat = contour.reshape(-1, 2)
            else:
                # skimage contour
                contour_flat = np.array(contour).reshape(-1, 2)
            
            # Skip very small contours
            if len(contour_flat) < 4:
                continue
                
            try:
                area = cv2.contourArea(contour_flat.reshape(-1, 1, 2))
                if area > largest_area:
                    largest_area = area
                    largest_contour_idx = i
            except:
                continue
        
        if largest_area > 0:
            largest_contour = contours[largest_contour_idx]
            if isinstance(largest_contour, np.ndarray) and len(largest_contour.shape) == 3:
                largest_contour = largest_contour.reshape(-1, 2)
            
            # Simplify using Douglas-Peucker algorithm
            try:
                epsilon = 0.01 * cv2.arcLength(largest_contour.reshape(-1, 1, 2), True)
                approx = cv2.approxPolyDP(largest_contour.reshape(-1, 1, 2), epsilon, True)
                approx = approx.reshape(-1, 2)
                
                if len(approx) >= 3:
                    # Directly create a polygon without Shapely
                    flattened = approx.flatten().tolist()
                    if len(flattened) >= 6:  # At least 3 points (x,y pairs)
                        polygons.append(flattened)
                        if verbose:
                            print(f"Added simplified polygon with {len(flattened)//2} points")
            except Exception as e:
                if verbose:
                    print(f"Error in simplified approach: {str(e)}")
    
    # If we still have no valid polygons, create a simple bounding box as a last resort
    if len(polygons) == 0:
        if verbose:
            print("No valid polygons created, falling back to bounding box")
        
        # Find bounding box of the mask
        y_indices, x_indices = np.where(binary_mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = float(np.min(x_indices)), float(np.max(x_indices))
            y_min, y_max = float(np.min(y_indices)), float(np.max(y_indices))
            
            # Create a simple rectangular polygon
            box_poly = [
                x_min, y_min,
                x_max, y_min,
                x_max, y_max,
                x_min, y_max
            ]
            
            polygons.append(box_poly)
            if verbose:
                print(f"Created rectangular polygon from mask bounds")
    
    # Convert polygons to COCO format
    coco_polygons = []
    for poly in polygons:
        # Check if this is already a flattened list
        if isinstance(poly, list) and all(isinstance(x, (int, float)) for x in poly):
            flattened = poly
        else:
            # Flatten polygon coordinates to [x1, y1, x2, y2, ...]
            flattened = np.array(poly).flatten().tolist()
        
        # COCO requires at least 6 points (3 x,y pairs)
        if len(flattened) >= 6:
            coco_polygons.append(flattened)
            if verbose:
                print(f"Added valid COCO polygon with {len(flattened)} coordinates")
        else:
            if verbose:
                print(f"Polygon has too few coordinates: {len(flattened)}")
    
    if len(coco_polygons) == 0 and verbose:
        print("Warning: No valid polygons created from mask")
    
    return coco_polygons

def get_object_id_mapping(xml_path, mask_path, categories_dict, verbose=False, seg_type='instance'):
    """Map object IDs in mask to object names in XML"""
    if verbose:
        print(f"Mapping objects between {os.path.basename(xml_path)} and {os.path.basename(mask_path)}")
    
    # Parse XML to get object bounding boxes
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        category_id = categories_dict[name]
        
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'category_id': category_id,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    # Load mask and get unique object IDs
    mask_img = Image.open(mask_path)
    mask_img = mask_img.convert("P")
    mask = np.array(mask_img)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids > 0]  # Remove background
    
    if verbose:
        print(f"Found {len(objects)} objects in XML and {len(obj_ids)} unique object IDs in mask")
        print(f"Unique mask values: {obj_ids}")
    
    # Create masks for each object ID
    sub_masks = create_sub_masks(mask_path, verbose, seg_type)
    
    # Different handling for semantic vs instance segmentation
    object_mapping = {}
    
    if seg_type == 'semantic':
        if verbose:
            print(f"Using semantic segmentation approach")
            
        # For semantic segmentation, we need to identify which class each mask ID corresponds to
        # In VOC, there's a specific mapping of pixel values to classes
        # We'll need to create a separate annotation for each class in the mask
        
        # Map each category to its unique mask value
        # This is the VOC semantic segmentation color mapping
        # Reference: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html
        voc_class_to_id = {
            'background': 0,
            'aeroplane': 1,
            'bicycle': 2,
            'bird': 3,
            'boat': 4,
            'bottle': 5,
            'bus': 6,
            'car': 7,
            'cat': 8,
            'chair': 9,
            'cow': 10,
            'diningtable': 11,
            'dog': 12,
            'horse': 13,
            'motorbike': 14,
            'person': 15,
            'pottedplant': 16,
            'sheep': 17,
            'sofa': 18,
            'train': 19,
            'tvmonitor': 20
        }
        
        # Reverse mapping: from ID to class name
        id_to_class = {v: k for k, v in voc_class_to_id.items() if v > 0}  # Skip background
        
        if verbose:
            print(f"VOC semantic segmentation class mapping: {id_to_class}")
            
        # Process each unique object ID in the mask
        for obj_id in obj_ids:
            if obj_id in id_to_class:
                class_name = id_to_class[obj_id]
                if class_name in categories_dict:
                    # Create a synthetic object for this class
                    category_id = categories_dict[class_name]
                    
                    # Find XML object with matching class if available
                    matching_obj = None
                    for obj in objects:
                        if obj['name'] == class_name:
                            matching_obj = obj
                            break
                    
                    if matching_obj:
                        object_mapping[obj_id] = matching_obj
                        if verbose:
                            print(f"Mapped semantic mask ID {obj_id} to class {class_name} with category ID {category_id}")
                    else:
                        # Create a synthetic object if no XML object exists
                        synthetic_obj = {
                            'name': class_name,
                            'category_id': category_id,
                            'bbox': [0, 0, 0, 0]  # Will be calculated from mask
                        }
                        object_mapping[obj_id] = synthetic_obj
                        if verbose:
                            print(f"Created synthetic object for class {class_name} with mask ID {obj_id}")
                else:
                    if verbose:
                        print(f"Warning: Class {class_name} not found in categories dictionary")
            else:
                if verbose:
                    print(f"Warning: Mask ID {obj_id} not found in VOC class mapping")
    else:
        # For instance segmentation, keep existing logic
        if verbose:
            print(f"Using instance segmentation mapping: mask object ID → XML object")
            
        # Try to match each mask object with an XML object based on their overlap
        for obj_id in obj_ids:
            sub_mask = sub_masks[obj_id]
            
            # Find bounding box of this mask
            y_indices, x_indices = np.where(sub_mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Find best matching XML object based on IoU
            best_iou = 0
            best_obj = None
            
            for obj in objects:
                xml_bbox = obj['bbox']
                
                # Calculate intersection
                inter_x_min = max(x_min, xml_bbox[0])
                inter_y_min = max(y_min, xml_bbox[1])
                inter_x_max = min(x_max, xml_bbox[2])
                inter_y_max = min(y_max, xml_bbox[3])
                
                if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
                    continue
                    
                inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                
                # Calculate union
                mask_area = (x_max - x_min) * (y_max - y_min)
                xml_area = (xml_bbox[2] - xml_bbox[0]) * (xml_bbox[3] - xml_bbox[1])
                union_area = mask_area + xml_area - inter_area
                
                # Calculate IoU
                iou = inter_area / union_area if union_area > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_obj = obj
            
            if best_obj and best_iou > 0.5:  # Only map if IoU is good enough
                object_mapping[obj_id] = best_obj
                if verbose:
                    print(f"Mapped object ID {obj_id} to {best_obj['name']} with IoU {best_iou:.2f}")
    
    return object_mapping, sub_masks

def parse_voc_xml(xml_path, segmask_path, categories, verbose=False, simplify_tolerance=0.5, seg_type='instance'):
    """Parse VOC XML annotation file and convert to COCO format"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    if verbose:
        print(f"Processing {filename}")
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': os.path.splitext(filename)[0]  # Use filename without extension as ID
    }
    
    annotations = []
    
    # Process segmentation mask if available
    if segmask_path and os.path.exists(segmask_path):
        if verbose:
            print(f"Processing segmentation mask: {segmask_path}")
        
        # Map object IDs in mask to object names in XML
        object_mapping, sub_masks = get_object_id_mapping(xml_path, segmask_path, categories, verbose, seg_type)
        
        if verbose:
            print(f"Found {len(object_mapping)} mapped objects in {filename}")
        
        # Create annotations with segmentation
        for obj_id, obj in object_mapping.items():
            sub_mask = sub_masks[obj_id]
            
            # Convert mask to polygon(s)
            segmentation = create_polygon_from_mask(sub_mask, tolerance=simplify_tolerance, verbose=verbose)
            
            if not segmentation:  # Skip if no valid polygon
                if verbose:
                    print(f"No valid polygon for object {obj_id} in {filename}, using bounding box instead")
                
                # Fall back to bounding box to create a simple rectangular polygon
                bndbox = obj['bbox']
                xmin, ymin, xmax, ymax = bndbox
                
                # Create a rectangular polygon from the bounding box
                polygon = [
                    xmin, ymin,
                    xmax, ymin,
                    xmax, ymax,
                    xmin, ymax
                ]
                
                segmentation = [polygon]
                if verbose:
                    print(f"Created rectangular polygon from bounding box")
            
            # Calculate bounding box and area
            y_indices, x_indices = np.where(sub_mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                if verbose:
                    print(f"Empty mask for object {obj_id} in {filename}, skipping")
                continue
                
            x_min, x_max = float(np.min(x_indices)), float(np.max(x_indices))
            y_min, y_max = float(np.min(y_indices)), float(np.max(y_indices))
            
            width = x_max - x_min
            height = y_max - y_min
            
            # Create annotation
            annotation = {
                'segmentation': segmentation,
                'area': float(np.sum(sub_mask)),  # Area is the number of pixels
                'iscrowd': 0,
                'image_id': image_info['id'],
                'bbox': [x_min, y_min, width, height],
                'category_id': obj['category_id'],
                'id': f"{image_info['id']}_{obj_id}"  # Unique annotation ID
            }
            
            if verbose:
                print(f"Added annotation for object {obj_id} with {len(segmentation)} polygons")
                # Count number of points in each polygon
                points_info = [f"{len(poly)//2} points" for poly in segmentation]
                print(f"Polygon details: {', '.join(points_info)}")
            
            annotations.append(annotation)
    
    # If we don't have segmentation or no valid polygons, fall back to bounding boxes
    if not annotations:
        if verbose:
            print(f"No segmentation masks found for {filename}, using bounding boxes")
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            category_id = categories[name]
            
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            width = xmax - xmin
            height = ymax - ymin
            
            # Create rectangular polygon from bounding box
            polygon = [
                xmin, ymin,
                xmax, ymin,
                xmax, ymax,
                xmin, ymax
            ]
            
            annotation = {
                'segmentation': [polygon],  # Use rectangular polygon instead of empty list
                'area': width * height,
                'iscrowd': 0,
                'image_id': image_info['id'],
                'bbox': [xmin, ymin, width, height],
                'category_id': category_id,
                'id': f"{image_info['id']}_{len(annotations)}"  # Unique annotation ID
            }
            
            annotations.append(annotation)
            
            if verbose:
                print(f"Added bounding box annotation for {name}")
    
    return image_info, annotations

def voc_to_coco(voc_dir, output_dir, sets=None, verbose=False, seg_type='instance', simplify_tolerance=0.5):
    """Convert VOC dataset to COCO format
    
    Args:
        voc_dir: Path to VOC dataset directory
        output_dir: Output directory for COCO dataset
        sets: Dataset splits to convert (default: ['train', 'val', 'test'])
        verbose: Whether to print verbose output
        seg_type: Type of segmentation ('instance' or 'semantic')
            - 'instance': Uses SegmentationObject folder where each object has a unique ID
            - 'semantic': Uses SegmentationClass folder where pixels are labeled by class
        simplify_tolerance: Tolerance for polygon simplification (lower = more detailed polygons)
    """
    annotations_dir = os.path.join(voc_dir, 'Annotations')
    images_dir = os.path.join(voc_dir, 'JPEGImages')
    
    # Use the Segmentation imagesets directory if we're doing segmentation conversion
    if seg_type in ['instance', 'semantic']:
        seg_imagesets_dir = os.path.join(voc_dir, 'ImageSets', 'Segmentation')
        if os.path.exists(seg_imagesets_dir):
            imagesets_dir = seg_imagesets_dir
            if verbose:
                print(f"Using ImageSets/Segmentation directory for segmentation data")
        else:
            imagesets_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
            if verbose:
                print(f"Warning: ImageSets/Segmentation directory not found, falling back to ImageSets/Main")
    else:
        imagesets_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
    
    # Set segmentation directory based on segmentation type
    if seg_type == 'instance':
        segmentation_dir = os.path.join(voc_dir, 'SegmentationObject')
        if verbose:
            print(f"Using instance segmentation masks from SegmentationObject directory")
            print(f"This will create instance segmentation masks where each object is individually segmented")
    else:  # semantic
        segmentation_dir = os.path.join(voc_dir, 'SegmentationClass')
        if verbose:
            print(f"Using semantic segmentation masks from SegmentationClass directory")
            print(f"This will create semantic segmentation masks where each class has a unique color")
    
    # Check if segmentation directory exists
    has_segmentation = os.path.isdir(segmentation_dir)
    if has_segmentation:
        if verbose:
            print(f"Found segmentation masks directory: {segmentation_dir}")
            print(f"Using polygon simplification tolerance: {simplify_tolerance} (lower = more detailed)")
            
            # Count number of PNG files
            seg_files = len([f for f in os.listdir(segmentation_dir) if f.endswith('.png')])
            print(f"Found {seg_files} segmentation mask files")
    else:
        if verbose:
            print("Segmentation masks directory not found. Only bounding boxes will be used.")
    
    if not sets:
        sets = ['train', 'val', 'test']
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_output_dir, exist_ok=True)
    annotations_output_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(annotations_output_dir, exist_ok=True)
    
    # Get all categories
    if verbose:
        print("Extracting categories...")
    categories_dict = get_categories(annotations_dir)
    if verbose:
        print(f"Found {len(categories_dict)} categories: {', '.join(categories_dict.keys())}")
    
    # Prepare categories list for COCO format
    categories = [{"id": cat_id, "name": cat_name, "supercategory": "none"} 
                 for cat_name, cat_id in categories_dict.items()]
    
    for dataset in sets:
        if verbose:
            print(f"\nProcessing {dataset} dataset...")
        
        # Skip if the dataset file doesn't exist
        dataset_file = os.path.join(imagesets_dir, f"{dataset}.txt")
        if not os.path.exists(dataset_file):
            if verbose:
                print(f"Warning: {dataset}.txt not found in {imagesets_dir}, skipping.")
            continue
        
        # Read image IDs
        with open(dataset_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines() if line.strip()]
        
        if verbose:
            print(f"Found {len(image_ids)} images in {dataset} dataset")
        
        # Create COCO dataset
        coco_output = {
            "info": {
                "description": f"VOC2012 {dataset} dataset converted to COCO format",
                "url": "http://host.robots.ox.ac.uk/pascal/VOC/",
                "version": "1.0",
                "year": 2012,
                "contributor": "Pascal VOC",
                "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [{"id": 1, "name": "Unknown License", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        # Create output directory for this dataset's images
        dataset_images_dir = os.path.join(images_output_dir, dataset)
        os.makedirs(dataset_images_dir, exist_ok=True)
        
        annotation_id = 1
        
        # Process each image
        for idx, img_id in enumerate(image_ids):
            if verbose and idx % 100 == 0:
                print(f"Processing image {idx+1}/{len(image_ids)}")
            
            xml_file = os.path.join(annotations_dir, f"{img_id}.xml")
            if not os.path.exists(xml_file):
                if verbose:
                    print(f"Warning: Annotation file {xml_file} not found, skipping.")
                continue
            
            # Check if segmentation mask exists
            segmask_path = None
            if has_segmentation:
                segmask_path = os.path.join(segmentation_dir, f"{img_id}.png")
                if not os.path.exists(segmask_path):
                    if verbose:
                        print(f"Warning: Segmentation mask not found for {img_id}")
                    segmask_path = None
            
            # Skip images without segmentation masks if we're specifically using the Segmentation imagesets
            if imagesets_dir.endswith('Segmentation') and segmask_path is None:
                if verbose:
                    print(f"Skipping {img_id} - no segmentation mask found and using Segmentation imagesets")
                continue
            
            # Parse annotation
            image_info, annotations = parse_voc_xml(xml_file, segmask_path, categories_dict, 
                                                    verbose=verbose, simplify_tolerance=simplify_tolerance, seg_type=seg_type)
            
            # Update annotation IDs
            for ann in annotations:
                ann['id'] = annotation_id
                annotation_id += 1
            
            # Copy image to output directory
            image_path = os.path.join(images_dir, image_info['file_name'])
            if not os.path.exists(image_path):
                image_path = os.path.join(images_dir, f"{img_id}.jpg")  # Try alternative extension
                if not os.path.exists(image_path):
                    if verbose:
                        print(f"Warning: Image file for {img_id} not found, skipping.")
                    continue
                else:
                    image_info['file_name'] = f"{img_id}.jpg"
            
            # Add numeric image ID
            image_info['id'] = idx + 1
            # Update image IDs in annotations
            for ann in annotations:
                ann['image_id'] = image_info['id']
            
            # Copy image to output directory
            dest_path = os.path.join(dataset_images_dir, image_info['file_name'])
            shutil.copy2(image_path, dest_path)
            
            # Add to COCO dataset
            coco_output['images'].append(image_info)
            coco_output['annotations'].extend(annotations)
        
        # Write COCO JSON to file
        json_path = os.path.join(annotations_output_dir, f"instances_{dataset}.json")
        with open(json_path, 'w') as f:
            json.dump(coco_output, f, indent=2)
        
        if verbose:
            print(f"Created COCO dataset with {len(coco_output['images'])} images and {len(coco_output['annotations'])} annotations")
            print(f"Saved to {json_path}")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VOC dataset to COCO format")
    parser.add_argument("--voc-dir", type=str, required=True, help="Path to VOC dataset directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for COCO dataset")
    parser.add_argument("--sets", type=str, nargs="+", default=["train", "val", "test"], 
                        help="Dataset splits to convert (default: train val test)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--seg-type", type=str, choices=["instance", "semantic"], default="instance",
                        help="Type of segmentation: 'instance' uses SegmentationObject folder, 'semantic' uses SegmentationClass folder")
    parser.add_argument("--simplify-tolerance", type=float, default=0.5, 
                        help="Tolerance for polygon simplification (lower = more detailed polygons, default: 0.5)")
    
    args = parser.parse_args()
    
    # Check requirements before running
    if not check_requirements():
        print("Missing required dependencies. Please install them and try again.")
        exit(1)
    
    start_time = time.time()
    output_path = voc_to_coco(args.voc_dir, args.output_dir, args.sets, args.verbose, args.seg_type, args.simplify_tolerance)
    end_time = time.time()
    
    print(f"\nConversion completed in {end_time - start_time:.2f} seconds")
    print(f"COCO dataset saved to {output_path}") 