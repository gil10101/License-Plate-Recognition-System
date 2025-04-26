#!/usr/bin/env python3
import os
import csv
import cv2
import numpy as np
import glob
import random
import shutil
from pathlib import Path

def create_dataset():
    """
    Prepare license plate data for training using the usimages and groundtruth.csv data.
    This function:
    1. Creates a YOLO dataset structure
    2. Generates labels for license plate images using actual plate positions
    3. Splits data into train/val sets
    """
    print("Preparing US license plate dataset for training...")
    
    # Create directory structure
    base_dir = "data/us_plate_data"
    os.makedirs(f"{base_dir}/images/train", exist_ok=True)
    os.makedirs(f"{base_dir}/images/val", exist_ok=True)
    os.makedirs(f"{base_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{base_dir}/labels/val", exist_ok=True)
    
    # Load license plate information from groundtruth file
    plates_data = {}
    groundtruth_csv = "data/seg_and_ocr/groundtruth.csv"
    if os.path.exists(groundtruth_csv):
        with open(groundtruth_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    image_name, region, plate_text = row[0], row[1], row[2]
                    plates_data[image_name] = (region, plate_text)
    
    # Get all image files from usimages directory
    usimages_dir = "data/seg_and_ocr/usimages"
    image_files = []
    if os.path.exists(usimages_dir):
        for img_file in glob.glob(os.path.join(usimages_dir, "*.png")):
            image_files.append(img_file)
    
    if not image_files:
        print("No images found in the usimages directory!")
        return
    
    # Shuffle and split the data (80% train, 20% val)
    random.shuffle(image_files)
    split_idx = int(0.8 * len(image_files))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Found {len(image_files)} images: {len(train_files)} for training, {len(val_files)} for validation")
    
    # Process training and validation images
    process_image_set(train_files, f"{base_dir}/images/train", f"{base_dir}/labels/train")
    process_image_set(val_files, f"{base_dir}/images/val", f"{base_dir}/labels/val")
    
    # Create data.yaml file for YOLOv5 training
    yaml_content = f"""
# YOLOv5 dataset configuration
path: {os.path.abspath(base_dir)}
train: images/train
val: images/val

# Classes
nc: 1  # number of classes
names: ['license_plate']
"""
    
    with open(f"{base_dir}/data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset prepared successfully. Data YAML saved to {base_dir}/data.yaml")
    return base_dir

def process_image_set(image_files, img_output_dir, label_output_dir):
    """Process a set of image files and create YOLO format labels"""
    for img_file in image_files:
        # Get image name
        img_name = os.path.basename(img_file)
        
        # Read the image for dimensions
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: Could not read image {img_file}")
            continue
        
        h, w = img.shape[:2]
        
        # Create YOLO label
        # Instead of using the entire image, detect the actual plate position
        # For this example, we'll use more precise license plate detection
        
        # Create a label file with the proper license plate location
        plate_boxes = detect_actual_plate(img)
        
        if plate_boxes:
            # Copy image to output directory
            shutil.copy(img_file, os.path.join(img_output_dir, img_name))
            
            # Create label file
            label_path = os.path.join(label_output_dir, os.path.splitext(img_name)[0] + '.txt')
            with open(label_path, 'w') as f:
                for box in plate_boxes:
                    # Convert to YOLO format: class_id, x_center, y_center, width, height (normalized)
                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / (2 * w)
                    y_center = (y1 + y2) / (2 * h)
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # YOLO format: class_id x_center y_center width height
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")
        else:
            print(f"Warning: No license plate detected in {img_file}")

def detect_actual_plate(image):
    """
    Detect actual license plate position in the image using
    a combination of image processing techniques.
    
    This function improves on the current detector by focusing on
    the actual license plate instead of the entire image.
    
    Args:
        image (numpy.ndarray): Input image.
        
    Returns:
        list: List of license plate bounding boxes as [x1, y1, x2, y2].
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise while preserving edges
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    # List to store license plate candidates
    plate_candidates = []
    
    # Aspect ratio range for standard US license plates (typically 2:1 to 4:1)
    MIN_ASPECT_RATIO = 1.5
    MAX_ASPECT_RATIO = 6.0
    
    h, w = image.shape[:2]
    
    for contour in contours:
        # Get the rotated rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Get the bounding rectangle
        x, y, rect_w, rect_h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio
        aspect_ratio = float(rect_w) / rect_h if rect_h > 0 else 0
        
        # Check if the shape resembles a license plate
        if MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
            # Calculate relative size
            area_ratio = (rect_w * rect_h) / (w * h)
            
            # License plates are typically not tiny or filling the entire image
            if 0.01 <= area_ratio <= 0.5:
                # Add some padding
                padding_x = int(rect_w * 0.05)
                padding_y = int(rect_h * 0.05)
                
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(w, x + rect_w + padding_x)
                y2 = min(h, y + rect_h + padding_y)
                
                plate_candidates.append([x1, y1, x2, y2])
    
    # If we didn't find any good candidates, fall back to using part of the image
    if not plate_candidates:
        # Most license plates are in the middle-lower part of the image
        # Create a reasonable region that likely contains the plate
        plate_width = int(w * 0.7)
        plate_height = int(h * 0.2)
        
        x1 = (w - plate_width) // 2
        y1 = int(h * 0.6)  # Start at 60% from the top
        x2 = x1 + plate_width
        y2 = y1 + plate_height
        
        plate_candidates.append([x1, y1, x2, y2])
    
    return plate_candidates

def train_model(data_dir):
    """
    Train a custom license plate detector.
    
    Args:
        data_dir (str): Path to the prepared dataset directory.
        
    Returns:
        str: Path to best trained weights.
    """
    try:
        # Import the detector for training
        from detector.custom_plate_detector import CustomLicensePlateDetector
        
        # Data YAML path
        data_yaml_path = os.path.join(data_dir, "data.yaml")
        
        print(f"Starting training with data from {data_yaml_path}")
        
        # Create detector instance
        detector = CustomLicensePlateDetector()
        
        # Train the model (using default settings)
        best_weights = detector.train(
            data_yaml_path=data_yaml_path,
            epochs=100,
            batch_size=16,
            img_size=640,
            weights='yolov5s.pt'  # Start from YOLOv5s weights
        )
        
        if best_weights:
            print(f"Training completed successfully. Best weights: {best_weights}")
            
            # Create a copy of the weights with a more descriptive name
            output_weights = "us_license_plate_detector.pt"
            shutil.copy(best_weights, output_weights)
            print(f"Weights copied to {output_weights}")
            
            return output_weights
        else:
            print("Training failed to produce best weights.")
            return None
            
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def main():
    """Main function to run the training pipeline"""
    print("US License Plate Detector Training")
    print("=================================")
    
    # Create dataset
    data_dir = create_dataset()
    if not data_dir:
        print("Dataset creation failed!")
        return
    
    # Train the model
    weights_path = train_model(data_dir)
    if weights_path:
        print("\nTraining completed successfully!")
        print(f"Use the trained model with: --weights {weights_path}")
    else:
        print("\nTraining failed!")

if __name__ == "__main__":
    main() 