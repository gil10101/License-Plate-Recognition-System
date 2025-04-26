#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
from pathlib import Path

def main():
    """
    Train a custom license plate detector using the YOLOv5 architecture.
    This script assumes the dataset has already been prepared by running prepare_dataset.py
    """
    # Force CUDA to be enabled if available
    import torch
    if torch.cuda.is_available():
        print("CUDA is available. Enabling GPU training.")
        torch.cuda.set_device(0)  # Use the first GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
    else:
        print("WARNING: CUDA is not available. Training will be slow on CPU.")
    
    parser = argparse.ArgumentParser(description='Train a custom license plate detector.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Initial weights for training')
    args = parser.parse_args()
    
    print("License Plate Detector Training")
    print("===============================")
    
    # Check if dataset is prepared
    data_yaml_path = 'data/license_plate_data/dataset.yaml'
    if not os.path.exists(data_yaml_path):
        print(f"Error: Data YAML file not found at {data_yaml_path}")
        print("Please run prepare_dataset.py first to prepare the dataset.")
        sys.exit(1)
    
    # Train the detector
    print("\nTraining license plate detector...")
    try:
        from detector.custom_plate_detector import CustomLicensePlateDetector
        
        detector = CustomLicensePlateDetector()
        weights_path = detector.train(
            data_yaml_path=data_yaml_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            weights=args.weights
        )
        
        if not weights_path:
            print("Error: Training failed to produce weights")
            sys.exit(1)
        
        # Copy best weights to a standard location
        standard_weights_path = 'license_plate_model.pt'
        shutil.copy(weights_path, standard_weights_path)
        
        print(f"Training completed successfully. Model saved to {standard_weights_path}")
    except Exception as e:
        print(f"Error training detector: {e}")
        sys.exit(1)
    
    print("\nLicense plate detector training process completed.")
    print(f"You can now use the trained model from {standard_weights_path}")

if __name__ == "__main__":
    main() 