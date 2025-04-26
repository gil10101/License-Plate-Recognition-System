#!/usr/bin/env python3
import os
import sys
import csv
import glob
import argparse
from improved_license_recognition import ImprovedLicensePlateRecognition

def load_groundtruth(csv_path):
    """
    Load license plate ground truth from CSV file
    
    Args:
        csv_path (str): Path to groundtruth CSV file
        
    Returns:
        dict: Dictionary mapping image filename to expected license plate text
    """
    groundtruth = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                image_name, region, plate_text = row[0], row[1], row[2]
                groundtruth[image_name] = plate_text
    
    return groundtruth

def calculate_accuracy(predictions, groundtruth, verbose=False):
    """
    Calculate accuracy of license plate recognition
    
    Args:
        predictions (dict): Dictionary of {image_name: predicted_text}
        groundtruth (dict): Dictionary of {image_name: expected_text}
        verbose (bool): Whether to print detailed results
        
    Returns:
        tuple: (accuracy, exact_matches, total_tests)
    """
    total = 0
    exact_matches = 0
    
    # Store incorrect predictions
    incorrect = []
    
    for img_name, pred_text in predictions.items():
        if img_name in groundtruth:
            total += 1
            expected_text = groundtruth[img_name]
            
            # Convert to uppercase for case-insensitive comparison
            pred_upper = pred_text.upper()
            expected_upper = expected_text.upper()
            
            # Remove spaces and common special characters for comparison
            pred_clean = ''.join(c for c in pred_upper if c.isalnum())
            expected_clean = ''.join(c for c in expected_upper if c.isalnum())
            
            # Check for exact match
            if pred_clean == expected_clean:
                exact_matches += 1
            else:
                incorrect.append({
                    'image': img_name,
                    'predicted': pred_text,
                    'expected': expected_text
                })
    
    # Calculate accuracy
    accuracy = exact_matches / total if total > 0 else 0
    
    # Print detailed results if requested
    if verbose:
        print("\nDetailed Results:")
        print(f"Total: {total}")
        print(f"Exact Matches: {exact_matches}")
        print(f"Accuracy: {accuracy:.2%}")
        
        if incorrect:
            print("\nIncorrect Predictions:")
            for entry in incorrect:
                print(f"Image: {entry['image']}")
                print(f"  Predicted: '{entry['predicted']}'")
                print(f"  Expected: '{entry['expected']}'")
                print()
    
    return accuracy, exact_matches, total

def main():
    """Main function to test the improved license plate recognition"""
    parser = argparse.ArgumentParser(description='Test Improved License Plate Recognition')
    parser.add_argument('--detector', type=str, help='Path to detector weights')
    parser.add_argument('--ocr-model', type=str, help='Path to OCR model directory')
    parser.add_argument('--limit', type=int, help='Limit the number of images to test')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--images-dir', type=str, default='data/seg_and_ocr/usimages',
                       help='Directory containing test images')
    parser.add_argument('--groundtruth', type=str, default='data/seg_and_ocr/groundtruth.csv',
                       help='Path to groundtruth CSV file')
    args = parser.parse_args()
    
    # Check if image directory exists
    if not os.path.exists(args.images_dir):
        print(f"Error: Image directory not found at {args.images_dir}")
        sys.exit(1)
    
    # Check if groundtruth file exists
    if not os.path.exists(args.groundtruth):
        print(f"Error: Groundtruth file not found at {args.groundtruth}")
        sys.exit(1)
    
    # Load groundtruth
    print(f"Loading groundtruth from {args.groundtruth}...")
    groundtruth = load_groundtruth(args.groundtruth)
    print(f"Loaded {len(groundtruth)} groundtruth entries")
    
    # Create license plate recognition system
    print("Initializing improved license plate recognition system...")
    lpr = ImprovedLicensePlateRecognition(
        detector_weights=args.detector,
        ocr_model_dir=args.ocr_model
    )
    
    # Get test images
    image_files = glob.glob(os.path.join(args.images_dir, "*.png"))
    
    # Limit number of images if requested
    if args.limit and args.limit > 0 and args.limit < len(image_files):
        print(f"Limiting to {args.limit} images")
        image_files = image_files[:args.limit]
    
    print(f"Testing on {len(image_files)} images...")
    
    # Process each image
    predictions = {}
    
    for i, image_path in enumerate(image_files):
        img_name = os.path.basename(image_path)
        
        # Skip images that are not in the groundtruth
        if img_name not in groundtruth:
            print(f"Skipping {img_name} (not in groundtruth)")
            continue
        
        # Show progress
        print(f"Processing {i+1}/{len(image_files)}: {img_name}...", end="")
        
        # Recognize license plate
        text, confidence, _ = lpr.recognize(image_path)
        
        # Store prediction
        predictions[img_name] = text
        
        # Show result
        expected = groundtruth[img_name]
        match = "✓" if text.upper() == expected.upper() else "✗"
        print(f" {match} Predicted: '{text}', Expected: '{expected}', Confidence: {confidence:.2f}")
    
    # Calculate accuracy
    accuracy, exact_matches, total = calculate_accuracy(predictions, groundtruth, args.verbose)
    
    # Print summary
    print("\nSummary:")
    print(f"Total: {total}")
    print(f"Exact Matches: {exact_matches}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("image,predicted,expected,correct\n")
            for img_name, pred_text in predictions.items():
                if img_name in groundtruth:
                    expected = groundtruth[img_name]
                    correct = 1 if pred_text.upper() == expected.upper() else 0
                    f.write(f"{img_name},{pred_text},{expected},{correct}\n")
        
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 