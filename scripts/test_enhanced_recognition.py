#!/usr/bin/env python3
import os
import sys
import csv
import glob
import argparse
import cv2
import numpy as np
from enhanced_recognition import EnhancedLicensePlateRecognition

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
    Calculate multiple accuracy metrics for license plate recognition
    
    Args:
        predictions (dict): Dictionary of {image_name: predicted_text}
        groundtruth (dict): Dictionary of {image_name: expected_text}
        verbose (bool): Whether to print detailed results
        
    Returns:
        dict: Dictionary of accuracy metrics
    """
    total = 0
    exact_matches = 0
    char_correct = 0
    total_chars = 0
    
    # Store detailed results for analysis
    results = []
    
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
            exact_match = (pred_clean == expected_clean)
            if exact_match:
                exact_matches += 1
            
            # Calculate character-level accuracy
            # Use Levenshtein distance for better character comparison
            min_len = min(len(pred_clean), len(expected_clean))
            max_len = max(len(pred_clean), len(expected_clean))
            
            # Count matching characters
            for i in range(min_len):
                if pred_clean[i] == expected_clean[i]:
                    char_correct += 1
            
            total_chars += len(expected_clean)
            
            # Calculate similarity score
            similarity = 0
            if max_len > 0:
                similarity = 1 - (levenshtein_distance(pred_clean, expected_clean) / max_len)
            
            # Store detailed results
            results.append({
                'image': img_name,
                'predicted': pred_text,
                'expected': expected_text,
                'exact_match': exact_match,
                'similarity': similarity
            })
    
    # Calculate metrics
    metrics = {
        'total': total,
        'exact_matches': exact_matches,
        'exact_accuracy': exact_matches / total if total > 0 else 0,
        'char_accuracy': char_correct / total_chars if total_chars > 0 else 0,
        'avg_similarity': sum(r['similarity'] for r in results) / len(results) if results else 0
    }
    
    # Print detailed results if requested
    if verbose:
        print("\nDetailed Results:")
        print(f"Total images: {metrics['total']}")
        print(f"Exact matches: {metrics['exact_matches']}")
        print(f"Exact accuracy: {metrics['exact_accuracy']:.2%}")
        print(f"Character accuracy: {metrics['char_accuracy']:.2%}")
        print(f"Average similarity: {metrics['avg_similarity']:.2%}")
        
        print("\nExample predictions:")
        # Sort by similarity (worst first)
        sorted_results = sorted(results, key=lambda x: x['similarity'])
        for i, result in enumerate(sorted_results[:10]):
            print(f"{i+1}. Image: {result['image']}")
            print(f"   Predicted: '{result['predicted']}'")
            print(f"   Expected:  '{result['expected']}'")
            print(f"   Match: {'✓' if result['exact_match'] else '✗'}")
            print(f"   Similarity: {result['similarity']:.2f}")
    
    return metrics, results

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def main():
    """Main function to test the enhanced license plate recognition"""
    parser = argparse.ArgumentParser(description='Test Enhanced License Plate Recognition')
    parser.add_argument('--detector', type=str, help='Path to detector weights')
    parser.add_argument('--ocr-model', type=str, help='Path to OCR model directory')
    parser.add_argument('--limit', type=int, help='Limit the number of images to test')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--images-dir', type=str, default='data/seg_and_ocr/usimages',
                       help='Directory containing test images')
    parser.add_argument('--groundtruth', type=str, default='data/seg_and_ocr/groundtruth.csv',
                       help='Path to groundtruth CSV file')
    parser.add_argument('--debug', action='store_true', help='Save debug images')
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
    print("Initializing enhanced license plate recognition system...")
    lpr = EnhancedLicensePlateRecognition(
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
    confidences = {}
    
    for i, image_path in enumerate(image_files):
        img_name = os.path.basename(image_path)
        
        # Skip images that are not in the groundtruth
        if img_name not in groundtruth:
            print(f"Skipping {img_name} (not in groundtruth)")
            continue
        
        # Show progress
        print(f"Processing {i+1}/{len(image_files)}: {img_name}...", end="")
        
        # Recognize license plate
        text, confidence, _ = lpr.recognize(image_path, debug=args.debug)
        
        # Store prediction
        predictions[img_name] = text
        confidences[img_name] = confidence
        
        # Show result
        expected = groundtruth[img_name]
        match = "✓" if text.upper().replace(" ", "") == expected.upper().replace(" ", "") else "✗"
        print(f" {match} Predicted: '{text}', Expected: '{expected}', Confidence: {confidence:.2f}")
    
    # Calculate accuracy metrics
    metrics, results = calculate_accuracy(predictions, groundtruth, args.verbose)
    
    # Print summary
    print("\nSummary:")
    print(f"Total: {metrics['total']}")
    print(f"Exact Matches: {metrics['exact_matches']}")
    print(f"Exact Accuracy: {metrics['exact_accuracy']:.2%}")
    print(f"Character Accuracy: {metrics['char_accuracy']:.2%}")
    print(f"Average Similarity: {metrics['avg_similarity']:.2%}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'predicted', 'expected', 'match', 'similarity', 'confidence'])
            
            for result in results:
                writer.writerow([
                    result['image'],
                    result['predicted'],
                    result['expected'],
                    1 if result['exact_match'] else 0,
                    result['similarity'],
                    confidences[result['image']]
                ])
        
        print(f"Results saved to {args.output}")
    
    # Print accuracy improvement (if available)
    try:
        with open('results.csv', 'r') as f:
            prev_matches = 0
            prev_total = 0
            for line in f:
                prev_total += 1
                
            prev_accuracy = prev_matches / prev_total if prev_total > 0 else 0
            
            improvement = metrics['exact_accuracy'] - prev_accuracy
            print(f"\nAccuracy improvement: {improvement:.2%}")
    except:
        # Previous results not available
        pass

if __name__ == "__main__":
    main() 