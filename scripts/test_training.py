#!/usr/bin/env python3
"""
Test script for license plate detector training.
This script runs a quick test of the training functionality with a small number of epochs.
"""

import os
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test the license plate detector training pipeline')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    args = parser.parse_args()
    
    print("License Plate Detector Training Test")
    print("===================================")
    print(f"Testing with {args.epochs} epochs and batch size {args.batch_size}")
    
    # Step 1: Test data preparation
    print("\n[1/3] Testing data preparation...")
    start_time = time.time()
    try:
        sys.path.append('data')
        from data.prepare_license_data import create_yolo_dataset
        
        # Change to data directory
        original_dir = os.getcwd()
        os.chdir('data')
        create_yolo_dataset()
        os.chdir(original_dir)
        
        data_yaml_path = 'data/license_plate_data/dataset.yaml'
        if not os.path.exists(data_yaml_path):
            print(f"Error: Data YAML file not found at {data_yaml_path}")
            return False
        
        prep_time = time.time() - start_time
        print(f"Data preparation completed in {prep_time:.2f} seconds")
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return False
    
    # Step 2: Test detector loading
    print("\n[2/3] Testing detector loading...")
    start_time = time.time()
    try:
        from detector.custom_plate_detector import CustomLicensePlateDetector
        
        detector = CustomLicensePlateDetector()
        
        load_time = time.time() - start_time
        print(f"Detector loaded in {load_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading detector: {e}")
        return False
    
    # Step 3: Test training (with minimal epochs)
    print("\n[3/3] Testing training (quick test)...")
    start_time = time.time()
    try:
        weights_path = detector.train(
            data_yaml_path=data_yaml_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=320  # Use small image size for faster testing
        )
        
        if not weights_path:
            print("Error: Training did not produce weights")
            return False
        
        train_time = time.time() - start_time
        print(f"Training test completed in {train_time:.2f} seconds")
        print(f"Weights saved to: {weights_path}")
    except Exception as e:
        print(f"Error in training: {e}")
        return False
    
    # Success!
    total_time = prep_time + load_time + train_time
    print("\nTest completed successfully!")
    print(f"Total test time: {total_time:.2f} seconds")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 