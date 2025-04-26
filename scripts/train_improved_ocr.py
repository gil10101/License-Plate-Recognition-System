#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pytesseract
import random
import glob
import csv
from pathlib import Path
import argparse
from improved_license_recognition import ImprovedLicensePlateRecognition

def load_groundtruth(csv_path):
    """Load license plate ground truth from CSV file"""
    groundtruth = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                image_name, region, plate_text = row[0], row[1], row[2]
                groundtruth[image_name] = plate_text
    
    return groundtruth

def create_training_dataset(images_dir, groundtruth):
    """Create a training dataset with plate images and their texts"""
    dataset = {}
    img_files = glob.glob(os.path.join(images_dir, "*.png"))
    
    for img_path in img_files:
        img_name = os.path.basename(img_path)
        if img_name in groundtruth:
            dataset[img_path] = groundtruth[img_name]
    
    return dataset

def extract_plate_regions(dataset, output_dir="data/extracted_plates"):
    """Extract license plate regions from images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize license plate recognizer
    lpr = ImprovedLicensePlateRecognition()
    
    plate_dataset = {}
    failed_extractions = []
    
    for img_path, text in dataset.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Detect plate using improved detection
        plate_img = lpr._detect_plate(img, debug=False)
        
        if plate_img is not None:
            # Save the extracted plate
            img_name = os.path.basename(img_path)
            plate_path = os.path.join(output_dir, img_name)
            cv2.imwrite(plate_path, plate_img)
            
            # Add to dataset
            plate_dataset[plate_path] = text
        else:
            failed_extractions.append(img_path)
    
    print(f"Extracted {len(plate_dataset)} license plate regions")
    print(f"Failed to extract {len(failed_extractions)} plates")
    
    return plate_dataset

def generate_augmented_data(plate_dataset, output_dir="data/augmented_plates"):
    """Generate augmented versions of license plate images"""
    os.makedirs(output_dir, exist_ok=True)
    
    augmented_dataset = {}
    
    for img_path, text in plate_dataset.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Original image
        augmented_dataset[img_path] = text
        
        # Augmentation 1: Brightness variations
        for i, alpha in enumerate([0.8, 1.2]):
            bright_img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
            out_path = os.path.join(output_dir, f"{base_name}_bright{i}.png")
            cv2.imwrite(out_path, bright_img)
            augmented_dataset[out_path] = text
        
        # Augmentation 2: Contrast variations
        for i, contrast in enumerate([0.8, 1.2]):
            mean = np.mean(img)
            contrast_img = img.copy()
            contrast_img = (contrast_img - mean) * contrast + mean
            contrast_img = np.clip(contrast_img, 0, 255).astype(np.uint8)
            out_path = os.path.join(output_dir, f"{base_name}_contrast{i}.png")
            cv2.imwrite(out_path, contrast_img)
            augmented_dataset[out_path] = text
        
        # Augmentation 3: Small rotations
        for i, angle in enumerate([-5, 5]):
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            out_path = os.path.join(output_dir, f"{base_name}_rotate{i}.png")
            cv2.imwrite(out_path, rotated)
            augmented_dataset[out_path] = text
        
        # Augmentation 4: Add noise
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        noisy_img = cv2.add(img, noise)
        out_path = os.path.join(output_dir, f"{base_name}_noise.png")
        cv2.imwrite(out_path, noisy_img)
        augmented_dataset[out_path] = text
    
    print(f"Generated {len(augmented_dataset)} images with augmentation")
    return augmented_dataset

def create_character_dataset(plate_dataset, output_dir="data/char_dataset"):
    """Create a dataset of individual license plate characters"""
    os.makedirs(output_dir, exist_ok=True)
    
    char_count = 0
    
    for img_path, text in plate_dataset.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find character contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours left-to-right
        if contours:
            sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
            
            # Extract character positions (potentially)
            char_positions = []
            for contour in sorted_contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by size to exclude noise
                if h > img.shape[0] * 0.3 and w > 5:
                    char_positions.append((x, y, w, h))
            
            # Check if we have roughly the same number of contours as text characters
            # If not, character segmentation might be unreliable
            if 0.7 <= len(char_positions) / len(text.replace(" ", "")) <= 1.3:
                # Try to match characters to contours
                text_no_spaces = text.replace(" ", "").upper()
                
                for i, (x, y, w, h) in enumerate(char_positions):
                    if i < len(text_no_spaces):
                        char = text_no_spaces[i]
                        
                        # Extract character image
                        char_img = gray[y:y+h, x:x+w]
                        
                        # Ensure it's not too small
                        if char_img.size > 0 and char_img.shape[0] > 10 and char_img.shape[1] > 5:
                            # Save character
                            char_dir = os.path.join(output_dir, char)
                            os.makedirs(char_dir, exist_ok=True)
                            
                            out_path = os.path.join(char_dir, f"{char}_{char_count}.png")
                            cv2.imwrite(out_path, char_img)
                            char_count += 1
    
    print(f"Extracted {char_count} individual characters")
    return char_count

def optimize_tesseract_params(plate_dataset, output_file="optimized_params.txt"):
    """Find optimal Tesseract parameters for license plate recognition"""
    best_params = None
    best_accuracy = 0
    
    # Parameters to test
    psm_values = [6, 7, 8, 9, 10, 11, 12, 13]
    oem_values = [1, 3]
    whitelist_values = [
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ",
    ]
    
    # Test combinations
    results = []
    for psm in psm_values:
        for oem in oem_values:
            for whitelist in whitelist_values:
                config = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist={whitelist}"
                
                correct = 0
                total = 0
                
                for img_path, text in plate_dataset.items():
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Try to recognize
                    try:
                        pred_text = pytesseract.image_to_string(gray, config=config).strip()
                        
                        # Basic cleanup
                        pred_clean = ''.join(c for c in pred_text.upper() if c.isalnum())
                        expected_clean = ''.join(c for c in text.upper() if c.isalnum())
                        
                        if pred_clean == expected_clean:
                            correct += 1
                        
                        total += 1
                    except Exception:
                        continue
                
                accuracy = correct / total if total > 0 else 0
                results.append((config, accuracy))
                
                print(f"Config: {config}, Accuracy: {accuracy:.2%}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = config
    
    # Save results
    with open(output_file, 'w') as f:
        f.write(f"Best config: {best_params}, Accuracy: {best_accuracy:.2%}\n\n")
        
        # Sort all results by accuracy
        results.sort(key=lambda x: x[1], reverse=True)
        
        for config, accuracy in results:
            f.write(f"{config}: {accuracy:.2%}\n")
    
    print(f"Best configuration: {best_params}")
    print(f"Best accuracy: {best_accuracy:.2%}")
    
    return best_params

def create_custom_corrections(plate_dataset, output_file="char_corrections.json"):
    """Create custom character correction mappings"""
    import json
    
    correction_data = {}
    confusion_matrix = {}
    
    for img_path, text in plate_dataset.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try to recognize with default config
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        try:
            pred_text = pytesseract.image_to_string(gray, config=config).strip()
            
            # Clean up predictions and ground truth
            pred_clean = ''.join(c for c in pred_text.upper() if c.isalnum())
            expected_clean = ''.join(c for c in text.upper() if c.isalnum())
            
            # If lengths match, analyze character by character
            if len(pred_clean) == len(expected_clean) and len(pred_clean) > 0:
                for i in range(len(pred_clean)):
                    pred_char = pred_clean[i]
                    expected_char = expected_clean[i]
                    
                    if pred_char != expected_char:
                        # Add to confusion matrix
                        if pred_char not in confusion_matrix:
                            confusion_matrix[pred_char] = {}
                        
                        if expected_char not in confusion_matrix[pred_char]:
                            confusion_matrix[pred_char][expected_char] = 0
                        
                        confusion_matrix[pred_char][expected_char] += 1
        except Exception:
            continue
    
    # Create correction mappings based on confusion matrix
    for pred_char, corrections in confusion_matrix.items():
        if corrections:
            # Find the most common correction
            best_correction = max(corrections.items(), key=lambda x: x[1])
            expected_char, count = best_correction
            
            # Only add if it appears multiple times
            if count >= 3:
                correction_data[pred_char] = expected_char
    
    # Add common license plate confusions
    default_corrections = {
        '0': 'O', 'O': '0',
        '1': 'I', 'I': '1',
        '5': 'S', 'S': '5',
        '8': 'B', 'B': '8',
        'Z': '2', '2': 'Z',
        'D': '0', 'Q': '0',
    }
    
    # Merge with our findings
    for char, correction in default_corrections.items():
        if char not in correction_data:
            correction_data[char] = correction
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(correction_data, f, indent=4)
    
    print(f"Created character correction mappings: {len(correction_data)} entries")
    return correction_data

def main():
    parser = argparse.ArgumentParser(description='Train improved OCR for license plates')
    parser.add_argument('--images-dir', type=str, default='data/seg_and_ocr/usimages',
                        help='Directory containing training images')
    parser.add_argument('--groundtruth', type=str, default='data/seg_and_ocr/groundtruth.csv',
                        help='Path to groundtruth CSV file')
    parser.add_argument('--output-dir', type=str, default='models/ocr_model',
                        help='Output directory for model files')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ground truth
    print(f"Loading groundtruth from {args.groundtruth}...")
    groundtruth = load_groundtruth(args.groundtruth)
    print(f"Loaded {len(groundtruth)} groundtruth entries")
    
    # Create training dataset
    dataset = create_training_dataset(args.images_dir, groundtruth)
    print(f"Created dataset with {len(dataset)} images")
    
    # Extract plate regions
    plate_dataset = extract_plate_regions(dataset)
    
    # Generate augmented data
    augmented_dataset = generate_augmented_data(plate_dataset)
    
    # Create character dataset for fine-tuning
    create_character_dataset(plate_dataset)
    
    # Optimize Tesseract parameters
    best_params = optimize_tesseract_params(augmented_dataset, 
                                           os.path.join(args.output_dir, "optimized_params.txt"))
    
    # Create custom character corrections
    create_custom_corrections(augmented_dataset, 
                             os.path.join(args.output_dir, "char_corrections.json"))
    
    print("Training completed.")
    print(f"Model files saved to {args.output_dir}")

if __name__ == "__main__":
    main() 