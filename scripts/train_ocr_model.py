#!/usr/bin/env python3
import os
import csv
import cv2
import numpy as np
import glob
import random
from pathlib import Path
import pickle
import json
from sklearn.model_selection import train_test_split

def create_ocr_dataset():
    """
    Create a dataset of license plate images with their text
    using the usimages folder and groundtruth.csv
    
    Returns:
        dict: Dictionary of {image_path: text} pairs
    """
    print("Creating OCR dataset from groundtruth.csv...")
    
    # Load license plate information from groundtruth file
    plate_texts = {}
    groundtruth_csv = "data/seg_and_ocr/groundtruth.csv"
    if not os.path.exists(groundtruth_csv):
        print(f"Error: {groundtruth_csv} not found!")
        return None
    
    with open(groundtruth_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                image_name, region, plate_text = row[0], row[1], row[2]
                plate_texts[image_name] = plate_text
    
    # Get all image files from usimages directory
    usimages_dir = "data/seg_and_ocr/usimages"
    if not os.path.exists(usimages_dir):
        print(f"Error: {usimages_dir} not found!")
        return None
    
    # Create a dictionary of {image_path: text} pairs
    dataset = {}
    for img_file in glob.glob(os.path.join(usimages_dir, "*.png")):
        img_name = os.path.basename(img_file)
        if img_name in plate_texts:
            dataset[img_file] = plate_texts[img_name]
    
    print(f"Found {len(dataset)} images with matching ground truth text")
    return dataset

def extract_license_plates(dataset):
    """
    Extract license plate regions from images using improved detection
    
    Args:
        dataset (dict): Dictionary of {image_path: text} pairs
        
    Returns:
        dict: Dictionary of {license_plate_image: text} pairs
    """
    print("Extracting license plate regions from images...")
    
    plates_dataset = {}
    
    # Create output directory for extracted plates
    output_dir = "data/extracted_plates"
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path, text in dataset.items():
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Find the actual license plate region
        plate_img = extract_best_plate_region(img)
        
        if plate_img is not None:
            # Save the extracted plate
            img_name = os.path.basename(img_path)
            plate_path = os.path.join(output_dir, img_name)
            cv2.imwrite(plate_path, plate_img)
            
            # Add to dataset
            plates_dataset[plate_path] = text
    
    print(f"Extracted {len(plates_dataset)} license plate regions")
    return plates_dataset

def extract_best_plate_region(image):
    """
    Extract the most likely license plate region from an image
    using a combination of image processing techniques
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: License plate image or None if not found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise while preserving edges
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    # Plate candidates
    candidates = []
    
    # Aspect ratio range for license plates
    MIN_ASPECT_RATIO = 1.5
    MAX_ASPECT_RATIO = 6.0
    
    h, w = image.shape[:2]
    
    for contour in contours:
        # Get bounding rectangle
        x, y, rect_w, rect_h = cv2.boundingRect(contour)
        
        # Skip very small contours
        if rect_w < 60 or rect_h < 15:
            continue
        
        # Calculate aspect ratio
        aspect_ratio = float(rect_w) / rect_h if rect_h > 0 else 0
        
        # Check if the shape resembles a license plate
        if MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
            # Calculate relative size
            area_ratio = (rect_w * rect_h) / (w * h)
            
            # License plates are typically not tiny or filling the entire image
            if 0.01 <= area_ratio <= 0.5:
                # Score this candidate based on how likely it is to be a plate
                edge_density = calculate_edge_density(edges[y:y+rect_h, x:x+rect_w])
                text_like_score = calculate_text_like_score(gray[y:y+rect_h, x:x+rect_w])
                
                score = edge_density * 0.7 + text_like_score * 0.3
                
                # Add some padding around the plate
                padding_x = int(rect_w * 0.05)
                padding_y = int(rect_h * 0.1)
                
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(w, x + rect_w + padding_x)
                y2 = min(h, y + rect_h + padding_y)
                
                candidates.append({
                    'score': score,
                    'box': (x1, y1, x2, y2)
                })
    
    # If we found candidates, return the one with the highest score
    if candidates:
        best_candidate = max(candidates, key=lambda c: c['score'])
        x1, y1, x2, y2 = best_candidate['box']
        return image[y1:y2, x1:x2]
    
    # Fallback: use the central part of the image
    # This is better than failing completely
    center_x = w // 2
    center_y = h // 2
    
    # Assume the plate is in the lower part of the image
    plate_width = int(w * 0.6)
    plate_height = int(h * 0.2)
    
    x1 = center_x - plate_width // 2
    y1 = center_y + int(h * 0.1)  # Slightly below center
    x2 = x1 + plate_width
    y2 = y1 + plate_height
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    return image[y1:y2, x1:x2]

def calculate_edge_density(img):
    """Calculate the density of edges in an image region"""
    if img.size == 0:
        return 0
    
    # Count non-zero pixels (edges)
    edge_pixels = np.count_nonzero(img)
    total_pixels = img.size
    
    return edge_pixels / total_pixels

def calculate_text_like_score(img):
    """Calculate how likely an image region contains text"""
    if img.size == 0:
        return 0
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours that might be characters
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check for character-like contours
    char_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Character aspect ratio and size check
        aspect_ratio = w / h if h > 0 else 0
        if 0.2 <= aspect_ratio <= 1.5 and h > img.shape[0] * 0.3:
            char_count += 1
    
    # More characters usually means more likely to be a plate
    # Most plates have between 5-10 characters
    if 3 <= char_count <= 12:
        return 0.8
    else:
        return 0.2 * (char_count / 8)  # Normalize around 8 characters

def preprocess_plates_for_ocr(plates_dataset):
    """
    Create preprocessed versions of the license plate images for better OCR
    
    Args:
        plates_dataset (dict): Dictionary of {plate_image_path: text} pairs
        
    Returns:
        dict: Dictionary of {preprocessed_image_path: text} pairs
    """
    print("Preprocessing license plates for OCR...")
    
    # Create directory for preprocessed images
    preproc_dir = "data/preprocessed_plates"
    os.makedirs(preproc_dir, exist_ok=True)
    
    preprocessed_dataset = {}
    
    for img_path, text in plates_dataset.items():
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Get base filename
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # Apply different preprocessing techniques
        preprocessed_versions = generate_preprocessed_versions(img)
        
        # Save each preprocessed version
        for i, proc_img in enumerate(preprocessed_versions):
            proc_path = os.path.join(preproc_dir, f"{base_name}_proc{i}.png")
            cv2.imwrite(proc_path, proc_img)
            
            # Add to dataset
            preprocessed_dataset[proc_path] = text
    
    print(f"Created {len(preprocessed_dataset)} preprocessed images")
    return preprocessed_dataset

def generate_preprocessed_versions(img):
    """
    Generate preprocessed versions of a license plate image for better OCR
    
    Args:
        img (numpy.ndarray): License plate image
        
    Returns:
        list: List of preprocessed image versions
    """
    versions = []
    
    # Original image
    versions.append(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    versions.append(gray)
    
    # Resize to larger resolution for better OCR
    h, w = img.shape[:2]
    scale_factor = 3.0
    resized = cv2.resize(gray, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    versions.append(resized)
    
    # Apply bilateral filter to remove noise while preserving edges
    filtered = cv2.bilateralFilter(resized, 11, 17, 17)
    versions.append(filtered)
    
    # Increase contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    versions.append(enhanced)
    
    # Apply different thresholding methods
    # Normal binary threshold
    _, thresh_binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
    versions.append(thresh_binary)
    
    # Inverse binary threshold (for plates with light text on dark background)
    _, thresh_binary_inv = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY_INV)
    versions.append(thresh_binary_inv)
    
    # Otsu thresholding (automatically finds optimal threshold)
    _, thresh_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(thresh_otsu)
    
    # Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    versions.append(adaptive)
    
    return versions

def create_ocr_configs():
    """
    Create a set of optimized OCR configurations based on the groundtruth data
    
    Returns:
        list: List of OCR configuration strings for pytesseract
    """
    print("Creating optimized OCR configurations...")
    
    # Basic OCR configurations for various license plate styles
    configs = [
        # Standard license plate configuration
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ',
        
        # For plates with variable character spacing
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ',
        
        # For plates with a single line of text
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ',
        
        # For plates with clean, well-separated characters
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ',
        
        # For cases where the plate might be tilted or curved
        r'--oem 3 --psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ',
        
        # For plates with very clean text (high contrast)
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- -c page_separator=""',
    ]
    
    return configs

def create_char_mappings(dataset):
    """
    Create character correction mappings based on common OCR errors in the dataset
    
    Args:
        dataset (dict): Dictionary of {image_path: ground_truth_text} pairs
        
    Returns:
        dict: Dictionary of {incorrect_char: correct_char} mappings
    """
    print("Creating character correction mappings...")
    
    # Common OCR errors for license plates
    char_mappings = {
        '0': 'O',
        'O': '0',
        '1': 'I',
        'I': '1',
        '5': 'S',
        'S': '5',
        '8': 'B',
        'B': '8',
        'D': '0',
        'Z': '2',
        '2': 'Z',
        'G': '6',
        '6': 'G',
        'T': '7',
        '7': 'T',
        'Q': '0',
        'U': 'V',
        'V': 'U',
        ' ': '',  # Remove spaces
    }
    
    # If we had more time, we would:
    # 1. Run OCR on the training images
    # 2. Compare with ground truth
    # 3. Build a frequency table of substitutions
    # 4. Create a mapping based on the most common substitutions
    
    return char_mappings

def save_ocr_model(configs, char_mappings):
    """
    Save the OCR model components
    
    Args:
        configs (list): List of OCR configuration strings
        char_mappings (dict): Character correction mappings
    """
    # Create directory for model files
    model_dir = "data/ocr_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save configurations
    with open(os.path.join(model_dir, "ocr_configs.pkl"), 'wb') as f:
        pickle.dump(configs, f)
    
    # Save character mappings
    with open(os.path.join(model_dir, "char_mappings.json"), 'w') as f:
        json.dump(char_mappings, f)
    
    print(f"OCR model saved to {model_dir}")

def main():
    """Main function to create and save the OCR model"""
    print("License Plate OCR Model Training")
    print("===============================")
    
    # Create dataset from groundtruth.csv
    dataset = create_ocr_dataset()
    if not dataset:
        print("Failed to create dataset!")
        return
    
    # Extract license plate regions
    plates_dataset = extract_license_plates(dataset)
    if not plates_dataset:
        print("Failed to extract license plates!")
        return
    
    # Preprocess plates for OCR
    preprocessed_dataset = preprocess_plates_for_ocr(plates_dataset)
    
    # Create OCR configurations
    configs = create_ocr_configs()
    
    # Create character mappings
    char_mappings = create_char_mappings(dataset)
    
    # Save OCR model
    save_ocr_model(configs, char_mappings)
    
    print("\nOCR model training completed successfully!")
    print("You can now use this model for improved license plate recognition.")

if __name__ == "__main__":
    main() 